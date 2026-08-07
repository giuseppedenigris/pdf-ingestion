import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from src import captioning, filtering

DEFAULT_MAX_CONSECUTIVE_FAILURES = 5

OnProgress = Callable[[str], None] | None


class CaptionRunState:
    # Promotes a run of per-item captioning failures into CaptionerUnavailable —
    # catches "server is up but broken" (e.g. wrong model name, every request
    # fails) without hardcoding a retry storm into the corpus loop.
    def __init__(self, max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES) -> None:
        self._max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_consecutive_failures:
            raise captioning.CaptionerUnavailable(
                f"{self._consecutive_failures} consecutive per-item captioning "
                "failures; treating captioner as unavailable for this run."
            )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _judged(meta: dict) -> bool:
    return meta.get("retrieval_text") is not None or meta.get("filtered_out", False)


def _apply_hard_filter(meta: dict, reason: str | None) -> bool:
    if reason is None:
        return False
    if meta.get("filtered_out") and meta.get("filtered_reason") == reason:
        return False
    meta["filtered_out"] = True
    meta["filtered_reason"] = reason
    return True


def _apply_assessment(meta: dict, result: captioning.ContentAssessment) -> None:
    meta["filtered_out"] = not result.useful
    meta["filtered_reason"] = result.reason
    meta["retrieval_text"] = result.caption


def _caption_images(
    output_dir: Path,
    picture_ids: list[str],
    captioner: captioning.Captioner | None,
    state: CaptionRunState,
    on_progress: OnProgress = None,
) -> None:
    images_dir = output_dir / "images"
    meta_dir = images_dir / "metadata"
    metas = {pid: _load_json(meta_dir / f"{pid}.json") for pid in picture_ids}

    for pid, meta in metas.items():
        reason = filtering.image_filter_reason(meta["width"], meta["height"])
        if _apply_hard_filter(meta, reason):
            _write_json(meta_dir / f"{pid}.json", meta)

    if captioner is None:
        return

    total = len(picture_ids)
    resolved = 0

    def report(count: int) -> None:
        nonlocal resolved
        resolved += count
        if on_progress:
            on_progress(f"captioning images ({resolved}/{total})")

    filtered_ids = [pid for pid, meta in metas.items() if meta.get("filtered_out", False)]
    if filtered_ids:
        report(len(filtered_ids))

    eligible = [pid for pid, meta in metas.items() if not meta.get("filtered_out", False)]
    groups: dict[str, list[str]] = defaultdict(list)
    for pid in eligible:
        content_hash = hashlib.sha256((images_dir / f"{pid}.png").read_bytes()).hexdigest()
        groups[content_hash].append(pid)

    for members in groups.values():
        done_id = next((m for m in members if _judged(metas[m])), None)
        if done_id is None:
            representative = sorted(members)[0]
            try:
                result = captioning.assess_image(
                    captioner,
                    (images_dir / f"{representative}.png").read_bytes(),
                    metas[representative].get("caption"),
                )
            except captioning.CaptionerUnavailable:
                raise
            except Exception as exc:
                state.record_failure()
                print(f"  captioning failed for image {representative}: {exc}")
            else:
                state.record_success()
                _apply_assessment(metas[representative], result)
                _write_json(meta_dir / f"{representative}.json", metas[representative])
                done_id = representative

        if done_id is not None:
            for member_id in members:
                if member_id == done_id or _judged(metas[member_id]):
                    continue
                metas[member_id]["filtered_out"] = metas[done_id]["filtered_out"]
                metas[member_id]["filtered_reason"] = metas[done_id]["filtered_reason"]
                metas[member_id]["retrieval_text"] = metas[done_id]["retrieval_text"]
                _write_json(meta_dir / f"{member_id}.json", metas[member_id])

        report(len(members))


def _caption_tables(
    output_dir: Path,
    table_ids: list[str],
    captioner: captioning.Captioner | None,
    state: CaptionRunState,
    on_progress: OnProgress = None,
) -> None:
    tables_dir = output_dir / "tables"
    meta_dir = tables_dir / "metadata"
    total = len(table_ids)

    for index, tid in enumerate(table_ids, start=1):
        meta_path = meta_dir / f"{tid}.json"
        meta = _load_json(meta_path)
        reason = filtering.table_filter_reason(meta["num_rows"], meta["num_cols"])
        if _apply_hard_filter(meta, reason):
            _write_json(meta_path, meta)

        if captioner is None:
            continue

        if not _judged(meta):
            html = (tables_dir / f"{tid}.html").read_text(encoding="utf-8")
            try:
                result = captioning.assess_table(captioner, html, meta.get("caption"))
            except captioning.CaptionerUnavailable:
                raise
            except Exception as exc:
                state.record_failure()
                print(f"  captioning failed for table {tid}: {exc}")
            else:
                state.record_success()
                _apply_assessment(meta, result)
                _write_json(meta_path, meta)

        if on_progress:
            on_progress(f"captioning tables ({index}/{total})")


def enrich_document(
    output_dir: Path,
    stem: str,
    captioner: captioning.Captioner | None,
    state: CaptionRunState,
    on_progress: OnProgress = None,
) -> None:
    doc_meta = _load_json(output_dir / "metadata" / f"{stem}.json")
    _caption_images(output_dir, doc_meta["picture_ids"], captioner, state, on_progress)
    _caption_tables(output_dir, doc_meta["table_ids"], captioner, state, on_progress)


def needs_enrichment(output_dir: Path, stem: str) -> bool:
    # Metadata-only check (no PNG hashing): is any item still unjudged?
    doc_meta = _load_json(output_dir / "metadata" / f"{stem}.json")
    images_meta_dir = output_dir / "images" / "metadata"
    tables_meta_dir = output_dir / "tables" / "metadata"
    for pid in doc_meta["picture_ids"]:
        if not _judged(_load_json(images_meta_dir / f"{pid}.json")):
            return True
    for tid in doc_meta["table_ids"]:
        if not _judged(_load_json(tables_meta_dir / f"{tid}.json")):
            return True
    return False
