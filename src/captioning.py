import base64
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Only a hard ceiling against a genuinely hung request, nothing more elaborate:
# per-item failures are handled by the caller (src/enrich.py), not retried here.
DEFAULT_TIMEOUT = 1800.0


class CaptionerUnavailable(Exception):
    """Captioner unreachable or broken; stop calling it for the rest of this run."""


# Split into 3 single-field schemas: a compound one made small models leak
# justification text into "reason" even when useful=True (verified empirically).
class UsefulResult(BaseModel):
    useful: bool = Field(
        description=(
            "True if structurally informative and worth retrieving (diagram, "
            "schematic, wiring diagram, exploded parts view, flowchart, "
            "technical/dimensional drawing, chart or graph, table, cross-section "
            "view, process/step illustration, annotated screenshot). False if it "
            "carries no retrievable information (decorative element, logo, "
            "watermark, icon, page divider/rule, blank or near-blank, generic "
            "marketing photo, page furniture, noisy/unreadable content)."
        )
    )


class CaptionResult(BaseModel):
    caption: str = Field(
        description=(
            "A detailed description for retrieval: what it shows, labeled "
            "parts/values if visible, likely purpose in context."
        )
    )


class ReasonResult(BaseModel):
    reason: Literal[
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
    ] = Field(description="The single best label for why this image is not useful.")


class TableReasonResult(BaseModel):
    # Separate vocabulary from ReasonResult: labels like logo/watermark/icon
    # don't apply to tables, and using the image enum made the model default
    # to "logo" nonsensically for things like blank forms and TOCs.
    reason: Literal[
        "empty_form",
        "table_of_contents",
        "noisy",
        "other",
    ] = Field(description="The single best label for why this table is not useful.")


class ContentAssessment(BaseModel):
    # Aggregate result consumed by src/enrich.py, assembled from up to two
    # model calls (see assess_image/assess_table below) rather than one.
    useful: bool
    reason: str | None = None
    caption: str | None = None


@dataclass
class Captioner:
    useful: Runnable
    caption: Runnable
    reason: Runnable
    table_reason: Runnable


_USEFUL_IMAGE_PROMPT = """You are assessing an image extracted from a technical manual \
(marine engines, electrical systems, installation guides). Is it structurally useful \
content worth retrieving — a diagram, schematic, wiring diagram, exploded parts view, \
flowchart, technical/dimensional drawing, chart or graph, table, cross-section view, \
process/step illustration, or annotated screenshot — or is it decorative/noise — a \
logo, watermark, icon, divider, blank or near-blank image, generic marketing photo, \
page furniture, or unreadable content?"""

_CAPTION_IMAGE_PROMPT = """This image, extracted from a technical manual (marine \
engines, electrical systems, installation guides), was judged useful. Write a \
detailed description for retrieval: what it shows, labeled parts or values if \
visible, likely purpose."""

_USEFUL_TABLE_PROMPT = """You are assessing a table extracted from a technical manual. \
Does it contain real structured data worth retrieving, or is it degenerate/noise \
(layout artifact, near-empty, garbled)?

Table HTML:
{html}"""

_CAPTION_TABLE_PROMPT = """This table, extracted from a technical manual, was judged \
useful. Describe what data it contains — headers, rows, what it lets a reader look up.

Table HTML:
{html}"""

_REASON_PROMPT = """This content, extracted from a technical manual, was judged not \
useful for retrieval. Pick the single best label for why."""

_REASON_TABLE_PROMPT = """This table, extracted from a technical manual, was judged not \
useful for retrieval. Pick the single best label for why.

Table HTML:
{html}"""

_NATIVE_CAPTION_SUFFIX = '\n\nThe document\'s own caption for this element, if any, is: "{caption}".'


def _with_native_caption(prompt: str, native_caption: str | None) -> str:
    if native_caption:
        return prompt + _NATIVE_CAPTION_SUFFIX.format(caption=native_caption)
    return prompt


def build_captioner(url: str, model: str) -> Captioner:
    llm = ChatOllama(base_url=url, model=model, client_kwargs={"timeout": DEFAULT_TIMEOUT})
    return Captioner(
        useful=llm.with_structured_output(UsefulResult),
        caption=llm.with_structured_output(CaptionResult),
        reason=llm.with_structured_output(ReasonResult),
        table_reason=llm.with_structured_output(TableReasonResult),
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


def assess_image(captioner: Captioner, png_bytes: bytes, native_caption: str | None) -> ContentAssessment:
    b64 = base64.b64encode(png_bytes).decode("ascii")

    def image_message(text: str) -> list[HumanMessage]:
        return [
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            )
        ]

    useful_result: UsefulResult = _invoke(
        captioner.useful, image_message(_with_native_caption(_USEFUL_IMAGE_PROMPT, native_caption))
    )
    if not useful_result.useful:
        reason_result: ReasonResult = _invoke(captioner.reason, image_message(_REASON_PROMPT))
        return ContentAssessment(useful=False, reason=reason_result.reason, caption=None)

    caption_result: CaptionResult = _invoke(
        captioner.caption, image_message(_with_native_caption(_CAPTION_IMAGE_PROMPT, native_caption))
    )
    return ContentAssessment(useful=True, reason=None, caption=caption_result.caption)


def assess_table(captioner: Captioner, html: str, native_caption: str | None) -> ContentAssessment:
    def table_message(text: str) -> list[HumanMessage]:
        return [HumanMessage(content=[{"type": "text", "text": text}])]

    useful_result: UsefulResult = _invoke(
        captioner.useful,
        table_message(_with_native_caption(_USEFUL_TABLE_PROMPT.format(html=html), native_caption)),
    )
    if not useful_result.useful:
        reason_result: TableReasonResult = _invoke(
            captioner.table_reason, table_message(_REASON_TABLE_PROMPT.format(html=html))
        )
        return ContentAssessment(useful=False, reason=reason_result.reason, caption=None)

    caption_result: CaptionResult = _invoke(
        captioner.caption,
        table_message(_with_native_caption(_CAPTION_TABLE_PROMPT.format(html=html), native_caption)),
    )
    return ContentAssessment(useful=True, reason=None, caption=caption_result.caption)
