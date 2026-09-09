from dataclasses import dataclass
from io import BytesIO

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

_MODEL_NAME = "google/siglip-base-patch16-224"

# Zero-shot labels as natural-language prompts (SigLIP is trained on caption-like
# text, not bare category words). "other" has no coherent visual concept to
# target, so it's left out — this classifier only ever proposes "useful" or one
# of these concrete junk categories.
_LABEL_PROMPTS: dict[str, str] = {
    "useful": "a technical diagram, schematic, or chart",
    "decorative": "a decorative design element",
    "logo": "a company or product logo",
    "watermark": "a watermark overlaid on an image",
    "icon": "a small UI icon or symbol",
    "divider": "a plain divider line or rule",
    "blank": "a blank or empty image",
    "noisy": "an unreadable, garbled image",
    "page_furniture": "a page header or footer element",
    "generic_photo": "a generic marketing or lifestyle photo",
}
_LABELS = list(_LABEL_PROMPTS.keys())
_PROMPTS = list(_LABEL_PROMPTS.values())

# Calibrated against 19 hand-verified images (100% top-1 accuracy, weakest
# correct case at margin=0.078) plus a 120-image sample of the real corpus
# (p10=0.071, median=0.51). Set with headroom above the weakest known-correct
# case, not picked arbitrarily.
JUNK_MARGIN_THRESHOLD = 0.3

_model: AutoModel | None = None
_processor: AutoProcessor | None = None


def _get_model() -> tuple[AutoModel, AutoProcessor]:
    # Lazy singleton: only the first call pays for loading the checkpoint.
    global _model, _processor
    if _model is None:
        _model = AutoModel.from_pretrained(_MODEL_NAME, device_map="auto")
        _model.eval()
        _processor = AutoProcessor.from_pretrained(_MODEL_NAME)
    return _model, _processor


@dataclass
class VisualFilterResult:
    label: str
    confidence: float
    margin: float  # gap between top and second-best label's confidence


def classify(png_bytes: bytes) -> VisualFilterResult:
    model, processor = _get_model()
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    inputs = processor(text=_PROMPTS, images=image, padding="max_length", return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
    # SigLIP's own sigmoid/BCE score is an independent per-label match probability
    # and came out uniformly tiny on this corpus (max 0.12 over 120 real images,
    # verified empirically) — useless as a confidence signal. Softmax over our
    # fixed candidate set instead answers "how much better is the top label than
    # the rest of *these* options", which is what a margin threshold needs.
    probs = torch.softmax(logits, dim=-1)[0]
    ranked = probs.argsort(descending=True)
    top, second = ranked[0].item(), ranked[1].item()
    return VisualFilterResult(
        label=_LABELS[top],
        confidence=probs[top].item(),
        margin=(probs[top] - probs[second]).item(),
    )


def maybe_skip_vlm(png_bytes: bytes) -> str | None:
    # Only ever short-circuits on a confident non-"useful" call: SigLIP can't
    # write a caption, so a "useful" prediction always still needs the VLM.
    result = classify(png_bytes)
    if result.label != "useful" and result.margin > JUNK_MARGIN_THRESHOLD:
        return result.label
    return None
