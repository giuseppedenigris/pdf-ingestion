import json
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DoclingDocument, DocItemLabel, PictureItem
from PIL.Image import Image

from src.utils import bounding_box, make_id, page_numbers


def build_picture_id_map(doc: DoclingDocument, binary_hash: int) -> dict[str, str]:
    return {
        picture.self_ref: make_id(binary_hash, "image", [picture.self_ref])
        for picture in doc.pictures
    }


def extract_embedded_labels(picture: PictureItem, doc: DoclingDocument) -> list[str] | None:
    # Children with a label other than "caption" are text fragments inside the
    # image's own bounding box (e.g. labels inside a diagram), not real captions.
    labels: list[str] = []
    for ref in picture.children:
        child = ref.resolve(doc)
        if hasattr(child, "text") and child.label != DocItemLabel.CAPTION:
            labels.append(child.text)
    return labels or None


def build_picture_metadata(
    picture: PictureItem,
    doc: DoclingDocument,
    image: Image,
    picture_id: str,
    doc_id: str,
    section_path: list[str],
) -> dict[str, Any]:
    caption = picture.caption_text(doc) or None
    return {
        "id": picture_id,
        "type": "image",
        "doc_id": doc_id,
        "self_ref": picture.self_ref,
        "page_no": page_numbers(picture),
        "bbox": bounding_box(picture),
        "section_path": section_path,
        "width": image.width,
        "height": image.height,
        "caption": caption,
        "caption_source": "native" if caption else "none",
        "embedded_labels": extract_embedded_labels(picture, doc),
        "retrieval_text": None,
        "filtered_out": False,
        "filtered_reason": None,
        "content_file": f"{picture_id}.png",
    }


def write_picture_outputs(
    image: Image,
    metadata: dict[str, Any],
    images_dir: Path,
) -> None:
    picture_id = metadata["id"]
    # PNG instead of JPEG: many pictures are diagrams/screenshots with fine text,
    # where JPEG's lossy compression introduces damaging artifacts.
    image.save(images_dir / f"{picture_id}.png")
    (images_dir / "metadata" / f"{picture_id}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
