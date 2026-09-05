import base64
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Only a hard ceiling against a genuinely hung request, nothing more elaborate:
# per-item failures are handled below by enrich_document, not retried here.
DEFAULT_TIMEOUT = 1800.0

# Caption generations are capped so a weak/quantized model can't ramble into a
# repetition loop (observed empirically) and so cost stays bounded; ~150-200
# words is enough for a retrieval caption.
CAPTION_MAX_TOKENS = 256

DEFAULT_MAX_CONSECUTIVE_FAILURES = 5

OnProgress = Callable[[str], None] | None


class CaptionerUnavailable(Exception):
    """Captioner unreachable or broken; stop calling it for the rest of this run."""


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
            raise CaptionerUnavailable(
                f"{self._consecutive_failures} consecutive per-item captioning "
                "failures; treating captioner as unavailable for this run."
            )


# Two-phase captioning: phase 1 is always exactly one call returning a single
# label (never a separate useful-bool + reason pair), so a not-useful item
# costs one call, not two. Phase 2 (captioning) runs only for items labeled
# "useful". A single Literal field can't leak justification prose the way a
# compound useful+reason schema did (verified empirically, hence the earlier
# split) — there's no free-text slot for a small model to fill.
class ImageLabelResult(BaseModel):
    label: Literal[
        "useful",
        "decorative",
        "logo",
        "watermark",
        "icon",
        "divider",
        "blank",
        "noisy",
        "page_furniture",
        "generic_photo",
        "other",
    ] = Field(
        description=(
            "Single best label: 'useful' if structurally informative and worth "
            "retrieving (diagram, schematic, wiring diagram, exploded parts view, "
            "flowchart, technical/dimensional drawing, chart or graph, table, "
            "cross-section view, process/step illustration, annotated screenshot), "
            "otherwise the single best reason it is not."
        )
    )


class TableLabelResult(BaseModel):
    # Separate vocabulary from ImageLabelResult: labels like logo/watermark/icon
    # don't apply to tables, and the image enum made models default to "logo"
    # nonsensically for things like blank forms and TOCs.
    label: Literal["useful", "empty_form", "table_of_contents", "noisy", "other"] = Field(
        description=(
            "Single best label: 'useful' if the table holds real structured data "
            "worth retrieving, otherwise the single best reason it is not."
        )
    )


class CaptionResult(BaseModel):
    caption: str = Field(
        description=(
            "A concise description for retrieval: what it shows, labeled "
            "parts/values if visible, likely purpose in context."
        )
    )


class ContentAssessment(BaseModel):
    # Aggregate result consumed by enrich_document below, assembled from up to
    # two model calls (see assess_image/assess_table) rather than one.
    useful: bool
    reason: str | None = None
    caption: str | None = None


@dataclass
class Captioner:
    image_label: Runnable
    table_label: Runnable
    caption: Runnable


_IMAGE_LABEL_PROMPT = """You are triaging an image extracted from a technical or installation \
manual. Reply with exactly one label:

- useful: structurally informative and worth retrieving — diagram, schematic, wiring \
diagram, exploded parts view, flowchart, technical/dimensional drawing, chart or graph, \
table, cross-section view, process/step illustration, or annotated screenshot.
- decorative: stylistic element carrying no retrievable information.
- logo: a company or product logo.
- watermark: an overlaid watermark.
- icon: a small UI or symbolic icon.
- divider: a page divider or rule line.
- blank: blank or near-blank.
- noisy: unreadable or too degraded to interpret.
- page_furniture: header/footer/layout element, not content.
- generic_photo: generic marketing or lifestyle photo with no technical content.
- other: none of the above, but still not worth retrieving."""

_CAPTION_IMAGE_PROMPT = """This image is extracted from a marine equipment manual. Describe \
concisely in 1-4 sentences what it shows: what is depicted, and any labeled parts or values \
if visible. Describe only what is visible or legibly labeled — if a specific product type, \
part name, or number is not clearly readable, describe it generically rather than guessing \
what it is."""

_TABLE_LABEL_PROMPT = """You are triaging a table extracted from a technical manual. Reply \
with exactly one label:

- useful: contains real structured data worth retrieving.
- empty_form: a blank form or template with no filled-in data.
- table_of_contents: a table of contents or index.
- noisy: layout artifact, near-empty, or garbled.
- other: degenerate or not worth retrieving for another reason.

Table HTML:
{html}"""

_CAPTION_TABLE_PROMPT = """This table, extracted from a technical manual, was judged useful. \
Describe concisely in 2-4 sentences what data it contains — headers, rows, what it lets a \
reader look up.

Table HTML:
{html}"""

_NATIVE_CAPTION_SUFFIX = '\n\nThe document\'s own caption for this element, if any, is: "{caption}".'


def _with_native_caption(prompt: str, native_caption: str | None) -> str:
    if native_caption:
        return prompt + _NATIVE_CAPTION_SUFFIX.format(caption=native_caption)
    return prompt


def build_captioner(url: str, model: str) -> Captioner:
    label_llm = ChatOllama(base_url=url, model=model, client_kwargs={"timeout": DEFAULT_TIMEOUT})
    caption_llm = ChatOllama(
        base_url=url,
        model=model,
        client_kwargs={"timeout": DEFAULT_TIMEOUT},
        num_predict=CAPTION_MAX_TOKENS,
    )
    return Captioner(
        image_label=label_llm.with_structured_output(ImageLabelResult),
        table_label=label_llm.with_structured_output(TableLabelResult),
        caption=caption_llm.with_structured_output(CaptionResult),
    )


def _invoke(runnable: Runnable, messages: list[HumanMessage]):
    try:
        return runnable.invoke(messages)
    except ConnectionError as exc:
        # ollama's client wraps httpx.ConnectError (nothing listening) into the
        # builtin ConnectionError. Anything else (timeouts, bad responses,
        # structured-output parsing errors) is left to propagate as-is: those
        # are per-item failures for the caller to catch, not "server is down".
        raise CaptionerUnavailable(f"cannot reach captioner: {exc}") from exc


def _assess(
    label_runnable: Runnable,
    caption_runnable: Runnable,
    label_prompt: str,
    caption_prompt: str,
    build_messages: Callable[[str], list[HumanMessage]],
) -> ContentAssessment:
    label_result = _invoke(label_runnable, build_messages(label_prompt))
    if label_result.label != "useful":
        return ContentAssessment(useful=False, reason=label_result.label, caption=None)

    caption_result: CaptionResult = _invoke(caption_runnable, build_messages(caption_prompt))
    return ContentAssessment(useful=True, reason=None, caption=caption_result.caption)


def assess_image(captioner: Captioner, png_bytes: bytes, native_caption: str | None) -> ContentAssessment:
    b64 = base64.b64encode(png_bytes).decode("ascii")

    def build_messages(text: str) -> list[HumanMessage]:
        return [
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            )
        ]

    return _assess(
        captioner.image_label,
        captioner.caption,
        _with_native_caption(_IMAGE_LABEL_PROMPT, native_caption),
        _with_native_caption(_CAPTION_IMAGE_PROMPT, native_caption),
        build_messages,
    )


def assess_table(captioner: Captioner, html: str, native_caption: str | None) -> ContentAssessment:
    def build_messages(text: str) -> list[HumanMessage]:
        return [HumanMessage(content=[{"type": "text", "text": text}])]

    return _assess(
        captioner.table_label,
        captioner.caption,
        _with_native_caption(_TABLE_LABEL_PROMPT.format(html=html), native_caption),
        _with_native_caption(_CAPTION_TABLE_PROMPT.format(html=html), native_caption),
        build_messages,
    )


# --- Orchestration: hard pre-filter, dedup, per-document enrichment ---


def _image_filter_reason(width: int, height: int) -> str | None:
    if width <= 0 or height <= 0:
        return "degenerate_dimensions"
    return None


def _table_filter_reason(num_rows: int, num_cols: int) -> str | None:
    # Not <=1: a legitimate single-column table (e.g. a parts list) is not degenerate.
    if num_rows == 0 or num_cols == 0:
        return "empty_table"
    return None


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


def _apply_assessment(meta: dict, result: ContentAssessment) -> None:
    meta["filtered_out"] = not result.useful
    meta["filtered_reason"] = result.reason
    meta["retrieval_text"] = result.caption


def _caption_images(
    output_dir: Path,
    picture_ids: list[str],
    captioner: Captioner | None,
    state: CaptionRunState,
    on_progress: OnProgress = None,
) -> None:
    images_dir = output_dir / "images"
    meta_dir = images_dir / "metadata"
    metas = {pid: _load_json(meta_dir / f"{pid}.json") for pid in picture_ids}

    for pid, meta in metas.items():
        reason = _image_filter_reason(meta["width"], meta["height"])
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
            rep_meta = metas[representative]
            if on_progress:
                on_progress(f"captioning images ({resolved}/{total}) {rep_meta['width']}x{rep_meta['height']}")
            try:
                result = assess_image(
                    captioner,
                    (images_dir / f"{representative}.png").read_bytes(),
                    metas[representative].get("caption"),
                )
            except CaptionerUnavailable:
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
    captioner: Captioner | None,
    state: CaptionRunState,
    on_progress: OnProgress = None,
) -> None:
    tables_dir = output_dir / "tables"
    meta_dir = tables_dir / "metadata"
    total = len(table_ids)

    for index, tid in enumerate(table_ids, start=1):
        meta_path = meta_dir / f"{tid}.json"
        meta = _load_json(meta_path)
        reason = _table_filter_reason(meta["num_rows"], meta["num_cols"])
        if _apply_hard_filter(meta, reason):
            _write_json(meta_path, meta)

        if captioner is None:
            continue

        if not _judged(meta):
            html = (tables_dir / f"{tid}.html").read_text(encoding="utf-8")
            try:
                result = assess_table(captioner, html, meta.get("caption"))
            except CaptionerUnavailable:
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
    captioner: Captioner | None,
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
