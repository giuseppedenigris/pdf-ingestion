import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.visual_filter import maybe_skip_vlm

# Only a hard ceiling against a genuinely hung request, nothing more elaborate:
# per-item failures are handled below, not retried here.
DEFAULT_TIMEOUT = 1800.0

# Caps a weak model's rambling/repetition loop while leaving headroom above the
# ~150-200 words a caption needs, so a verbose-but-correct answer isn't truncated.
CAPTION_MAX_TOKENS = 512

# Rejects degenerate completions (e.g. "Table HTML content"); not in the prompt,
# so a lazy model can't just pad filler to clear the bar.
MIN_CAPTION_WORDS = 8

DEFAULT_MAX_CONSECUTIVE_FAILURES = 5

OnProgress = Callable[[str], None] | None


class CaptionerUnavailable(Exception):
    """Captioner unreachable or broken; stop calling it for the rest of this run."""


class CaptionRunState:
    # Promotes a run of per-item captioning failures into CaptionerUnavailable —
    # catches "server is up but broken" (e.g. wrong model name, every request
    # fails) without hardcoding a retry storm into the corpus loop. Images and
    # tables use separate independent instances of this: one model stream
    # being down should not disable the other.
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


class ContentAssessment(BaseModel):
    # Aggregate result consumed by caption_images/caption_tables below, assembled
    # from up to two model calls (see assess_image/assess_table) rather than one.
    useful: bool
    reason: str | None = None
    caption: str | None = None


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


def _with_context(
    prompt: str,
    native_caption: str | None,
    section_path: list[str] | None = None,
    surrounding_text: str | None = None,
) -> str:
    if section_path:
        prompt += f"\n\nThis element appears under the section: {' > '.join(section_path)}."
    if surrounding_text:
        prompt += f"\n\nSurrounding document text:\n{surrounding_text}"
    if native_caption:
        prompt += f'\n\nThe document\'s own caption for this element, if any, is: "{native_caption}".'
    return prompt


def _is_prose(block: str) -> bool:
    stripped = block.strip()
    return bool(stripped) and not stripped.startswith(("[[IMG:", "[[TABLE:", "#"))


def _surrounding_text(markdown_text: str, anchor: str, window: int = 1, search_limit: int = 5) -> str | None:
    # A neighboring block is often another anchor or a bare heading, not prose
    # (verified against real generated markdown) — walk outward past those to
    # find genuine surrounding text instead of grabbing whatever's adjacent.
    blocks = markdown_text.split("\n\n")
    anchor_idx = next((i for i, b in enumerate(blocks) if anchor in b), None)
    if anchor_idx is None:
        return None

    def collect(indices: range) -> list[str]:
        found = []
        for i in indices:
            if 0 <= i < len(blocks) and _is_prose(blocks[i]):
                found.append(blocks[i].strip())
                if len(found) >= window:
                    break
        return found

    before = collect(range(anchor_idx - 1, anchor_idx - 1 - search_limit, -1))
    after = collect(range(anchor_idx + 1, anchor_idx + 1 + search_limit))
    snippet = "\n\n".join(before[::-1] + after).strip()
    return snippet[:1000] if snippet else None


def _load_markdown(output_dir: Path, stem: str) -> str | None:
    md_path = output_dir / f"{stem}.md"
    return md_path.read_text(encoding="utf-8") if md_path.exists() else None


def _build_pair(ollama_url: str, model: str, label_schema: type[BaseModel]) -> tuple[Runnable, Runnable]:
    label_llm = ChatOllama(base_url=ollama_url, model=model, client_kwargs={"timeout": DEFAULT_TIMEOUT})
    caption_llm = ChatOllama(
        base_url=ollama_url,
        model=model,
        client_kwargs={"timeout": DEFAULT_TIMEOUT},
        num_predict=CAPTION_MAX_TOKENS,
    )
    # Label is categorical, structured output fits. Caption is free prose —
    # forcing a JSON/tool-call schema on that risks a minimal-effort shortcut.
    return label_llm.with_structured_output(label_schema), caption_llm


def build_image_captioner(ollama_url: str, model: str) -> tuple[Runnable, Runnable]:
    return _build_pair(ollama_url, model, ImageLabelResult)


def build_table_captioner(ollama_url: str, model: str) -> tuple[Runnable, Runnable]:
    return _build_pair(ollama_url, model, TableLabelResult)


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

    caption = _invoke(caption_runnable, build_messages(caption_prompt)).content.strip()
    if len(caption.split()) < MIN_CAPTION_WORDS:
        raise ValueError(f"caption too short ({len(caption.split())} words): {caption!r}")
    return ContentAssessment(useful=True, reason=None, caption=caption)


def assess_image(
    label_runnable: Runnable,
    caption_runnable: Runnable,
    png_bytes: bytes,
    native_caption: str | None,
    section_path: list[str] | None = None,
    surrounding_text: str | None = None,
) -> ContentAssessment:
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
        label_runnable,
        caption_runnable,
        _with_context(_IMAGE_LABEL_PROMPT, native_caption, section_path, surrounding_text),
        _with_context(_CAPTION_IMAGE_PROMPT, native_caption, section_path, surrounding_text),
        build_messages,
    )


def assess_table(
    label_runnable: Runnable,
    caption_runnable: Runnable,
    html: str,
    native_caption: str | None,
    section_path: list[str] | None = None,
    surrounding_text: str | None = None,
) -> ContentAssessment:
    def build_messages(text: str) -> list[HumanMessage]:
        return [HumanMessage(content=[{"type": "text", "text": text}])]

    return _assess(
        label_runnable,
        caption_runnable,
        _with_context(_TABLE_LABEL_PROMPT.format(html=html), native_caption, section_path, surrounding_text),
        _with_context(_CAPTION_TABLE_PROMPT.format(html=html), native_caption, section_path, surrounding_text),
        build_messages,
    )


# --- Orchestration: hard pre-filter, dedup, per-document captioning ---


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
    tagged_reason = f"(hard_filter) {reason}"
    if meta.get("filtered_out") and meta.get("filtered_reason") == tagged_reason:
        return False
    meta["filtered_out"] = True
    meta["filtered_reason"] = tagged_reason
    return True


def _apply_assessment(meta: dict, result: ContentAssessment) -> None:
    meta["filtered_out"] = not result.useful
    meta["filtered_reason"] = f"(vlm) {result.reason}" if result.reason is not None else None
    meta["retrieval_text"] = result.caption


def caption_images(
    output_dir: Path,
    stem: str,
    image_captioner: tuple[Runnable, Runnable] | None,
    state: CaptionRunState,
    on_progress: OnProgress = None,
) -> tuple[int, int, int, int]:
    """Returns (total, already_done, filtered_out, captioned) counts for this document."""
    doc_meta = _load_json(output_dir / "metadata" / f"{stem}.json")
    picture_ids = doc_meta["picture_ids"]
    total = len(picture_ids)

    images_dir = output_dir / "images"
    meta_dir = images_dir / "metadata"
    metas = {pid: _load_json(meta_dir / f"{pid}.json") for pid in picture_ids}

    already_done = sum(1 for meta in metas.values() if _judged(meta))
    pending = [pid for pid, meta in metas.items() if not _judged(meta)]

    for pid in pending:
        meta = metas[pid]
        reason = _image_filter_reason(meta["width"], meta["height"])
        if _apply_hard_filter(meta, reason):
            _write_json(meta_dir / f"{pid}.json", meta)

    # Cheap local classifier catches confident junk (logos, watermarks, blanks,
    # ...) before spending a VLM call on it. Runs regardless of whether a VLM
    # captioner is configured this run — it's local, no dependency on one.
    to_check = [pid for pid in pending if not metas[pid].get("filtered_out", False)]
    for checked, pid in enumerate(to_check, start=1):
        if on_progress:
            on_progress(f"checking images ({checked}/{len(to_check)})")
        meta = metas[pid]
        try:
            skip_reason = maybe_skip_vlm((images_dir / f"{pid}.png").read_bytes())
        except Exception:
            # Local classifier is an optimization, not a dependency: any
            # failure (bad image, model unavailable) just falls through to
            # the normal VLM judgment below instead of aborting the run.
            skip_reason = None
        if skip_reason is not None:
            meta["filtered_out"] = True
            meta["filtered_reason"] = f"(siglip) {skip_reason}"
            _write_json(meta_dir / f"{pid}.json", meta)

    if image_captioner is None:
        filtered_out = sum(1 for pid in pending if metas[pid].get("filtered_out", False))
        return total, already_done, filtered_out, 0

    label_runnable, caption_runnable = image_captioner
    markdown_text = _load_markdown(output_dir, stem)

    # Dedup groups only ever contain pending members: an already-done image
    # never joins one, so a group is either entirely new work or doesn't exist.
    to_caption = [pid for pid in pending if not metas[pid].get("filtered_out", False)]
    groups: dict[str, list[str]] = defaultdict(list)
    for pid in to_caption:
        content_hash = hashlib.sha256((images_dir / f"{pid}.png").read_bytes()).hexdigest()
        groups[content_hash].append(pid)

    for index, members in enumerate(groups.values(), start=1):
        representative = sorted(members)[0]
        rep_meta = metas[representative]
        if on_progress:
            on_progress(f"captioning images ({index}/{len(groups)}) {rep_meta['width']}x{rep_meta['height']}")
        surrounding = (
            _surrounding_text(markdown_text, f"[[IMG:{representative}]]") if markdown_text else None
        )
        try:
            result = assess_image(
                label_runnable,
                caption_runnable,
                (images_dir / f"{representative}.png").read_bytes(),
                rep_meta.get("caption"),
                rep_meta.get("section_path"),
                surrounding,
            )
        except CaptionerUnavailable:
            raise
        except Exception as exc:
            state.record_failure()
            print(f"  captioning failed for image {representative}: {exc}")
        else:
            state.record_success()
            _apply_assessment(rep_meta, result)
            _write_json(meta_dir / f"{representative}.json", rep_meta)
            for member_id in members:
                if member_id == representative:
                    continue
                metas[member_id]["filtered_out"] = rep_meta["filtered_out"]
                metas[member_id]["filtered_reason"] = rep_meta["filtered_reason"]
                metas[member_id]["retrieval_text"] = rep_meta["retrieval_text"]
                _write_json(meta_dir / f"{member_id}.json", metas[member_id])

    filtered_out = sum(1 for pid in pending if metas[pid].get("filtered_out", False))
    captioned = sum(1 for pid in pending if metas[pid].get("retrieval_text") is not None)
    return total, already_done, filtered_out, captioned


def caption_tables(
    output_dir: Path,
    stem: str,
    table_captioner: tuple[Runnable, Runnable] | None,
    state: CaptionRunState,
    on_progress: OnProgress = None,
) -> tuple[int, int, int, int]:
    """Returns (total, already_done, filtered_out, captioned) counts for this document."""
    doc_meta = _load_json(output_dir / "metadata" / f"{stem}.json")
    table_ids = doc_meta["table_ids"]
    total = len(table_ids)

    tables_dir = output_dir / "tables"
    meta_dir = tables_dir / "metadata"
    metas = {tid: _load_json(meta_dir / f"{tid}.json") for tid in table_ids}

    already_done = sum(1 for meta in metas.values() if _judged(meta))
    pending = [tid for tid, meta in metas.items() if not _judged(meta)]

    for tid in pending:
        meta = metas[tid]
        reason = _table_filter_reason(meta["num_rows"], meta["num_cols"])
        if _apply_hard_filter(meta, reason):
            _write_json(meta_dir / f"{tid}.json", meta)

    if table_captioner is None:
        filtered_out = sum(1 for tid in pending if metas[tid].get("filtered_out", False))
        return total, already_done, filtered_out, 0

    label_runnable, caption_runnable = table_captioner
    markdown_text = _load_markdown(output_dir, stem)

    to_caption = [tid for tid in pending if not metas[tid].get("filtered_out", False)]
    for index, tid in enumerate(to_caption, start=1):
        meta = metas[tid]
        html = (tables_dir / f"{tid}.html").read_text(encoding="utf-8")
        surrounding = _surrounding_text(markdown_text, f"[[TABLE:{tid}]]") if markdown_text else None
        if on_progress:
            on_progress(f"captioning tables ({index}/{len(to_caption)})")
        try:
            result = assess_table(
                label_runnable, caption_runnable, html, meta.get("caption"), meta.get("section_path"), surrounding
            )
        except CaptionerUnavailable:
            raise
        except Exception as exc:
            state.record_failure()
            print(f"  captioning failed for table {tid}: {exc}")
        else:
            state.record_success()
            _apply_assessment(meta, result)
            _write_json(meta_dir / f"{tid}.json", meta)

    filtered_out = sum(1 for tid in pending if metas[tid].get("filtered_out", False))
    captioned = sum(1 for tid in pending if metas[tid].get("retrieval_text") is not None)
    return total, already_done, filtered_out, captioned


def needs_image_captioning(output_dir: Path, stem: str) -> bool:
    doc_meta = _load_json(output_dir / "metadata" / f"{stem}.json")
    images_meta_dir = output_dir / "images" / "metadata"
    return any(not _judged(_load_json(images_meta_dir / f"{pid}.json")) for pid in doc_meta["picture_ids"])


def needs_table_captioning(output_dir: Path, stem: str) -> bool:
    doc_meta = _load_json(output_dir / "metadata" / f"{stem}.json")
    tables_meta_dir = output_dir / "tables" / "metadata"
    return any(not _judged(_load_json(tables_meta_dir / f"{tid}.json")) for tid in doc_meta["table_ids"])
