import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from docling_core.transforms.serializer.base import BaseTableSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownPictureSerializer,
)
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRefMode,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TitleItem,
)

from src import images as images_module
from src import tables as tables_module
from src.conversion import convert_pdf
from src.utils import make_id


def compute_section_paths(doc: DoclingDocument) -> dict[str, list[str]]:
    heading_by_level: dict[int, str] = {}
    section_path_by_ref: dict[str, list[str]] = {}

    for item, _level in doc.iterate_items(with_groups=True):
        if isinstance(item, TitleItem):
            heading_by_level = {0: item.text}
        elif isinstance(item, SectionHeaderItem):
            heading_by_level = {
                level: text for level, text in heading_by_level.items() if level < item.level
            }
            heading_by_level[item.level] = item.text
        else:
            self_ref = getattr(item, "self_ref", None)
            if self_ref is not None:
                section_path_by_ref[self_ref] = [
                    heading_by_level[level] for level in sorted(heading_by_level)
                ]

    return section_path_by_ref


class _AnchorTableSerializer(BaseTableSerializer):
    def __init__(self, id_map: dict[str, str]) -> None:
        self.id_map = id_map

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: Any,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        chunk_id = self.id_map.get(item.self_ref, item.self_ref)
        parts = [create_ser_result(text=f"[[TABLE:{chunk_id}]]", span_source=item)]
        # serialize_captions is mandatory: without it, captions linked to this
        # table via item.captions are silently dropped from the markdown output.
        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        if cap_res.text:
            parts.append(cap_res)
        return create_ser_result(text="\n".join(p.text for p in parts), span_source=parts)


class _AnchorPictureSerializer(MarkdownPictureSerializer):
    def __init__(self, id_map: dict[str, str]) -> None:
        super().__init__()
        self.id_map = id_map

    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: Any,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        chunk_id = self.id_map.get(item.self_ref, item.self_ref)
        parts = [create_ser_result(text=f"[[IMG:{chunk_id}]]", span_source=item)]
        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        if cap_res.text:
            parts.append(cap_res)
        return create_ser_result(text="\n".join(p.text for p in parts), span_source=parts)


def _render_markdown_with_anchors(
    doc: DoclingDocument,
    table_id_map: dict[str, str],
    picture_id_map: dict[str, str],
) -> str:
    serializer = MarkdownDocSerializer(
        doc=doc,
        table_serializer=_AnchorTableSerializer(table_id_map),
        picture_serializer=_AnchorPictureSerializer(picture_id_map),
    )
    return serializer.serialize().text


def _prepare_default_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    metadata_dir = output_dir / "metadata"
    tables_dir = output_dir / "tables"
    images_dir = output_dir / "images"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (images_dir / "metadata").mkdir(parents=True, exist_ok=True)
    return metadata_dir, tables_dir, images_dir


def _prepare_md_only_output_dirs(output_dir: Path) -> Path:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir


def _write_document_metadata(metadata_dir: Path, stem: str, metadata: dict[str, Any]) -> None:
    (metadata_dir / f"{stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_document_metadata(
    doc_id: str,
    pdf_path: Path,
    num_pages: int,
    stem: str,
    table_ids: list[str],
    picture_ids: list[str],
) -> dict:
    return {
        "doc_id": doc_id,
        "source_file": pdf_path.name,
        "num_pages": num_pages,
        "md_file": f"{stem}.md",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "table_ids": table_ids,
        "picture_ids": picture_ids,
    }


def process_pdf_default(converter: DocumentConverter, pdf_path: Path, output_dir: Path) -> None:
    doc = convert_pdf(converter, pdf_path)
    stem = pdf_path.stem
    binary_hash = doc.origin.binary_hash
    doc_id = make_id(binary_hash, "doc", [])

    table_id_map = tables_module.build_table_id_map(doc, binary_hash)
    picture_id_map = images_module.build_picture_id_map(doc, binary_hash)
    section_path_by_ref = compute_section_paths(doc)

    metadata_dir, tables_dir, images_dir = _prepare_default_output_dirs(output_dir)

    markdown = _render_markdown_with_anchors(doc, table_id_map, picture_id_map)
    (output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")

    table_ids: list[str] = []
    for table in doc.tables:
        table_id = table_id_map[table.self_ref]
        metadata = tables_module.build_table_metadata(
            table, doc, table_id, doc_id, section_path_by_ref.get(table.self_ref, [])
        )
        tables_module.write_table_outputs(table, doc, metadata, tables_dir)
        table_ids.append(table_id)

    picture_ids: list[str] = []
    for picture in doc.pictures:
        image = picture.get_image(doc)
        if image is None:
            continue
        picture_id = picture_id_map[picture.self_ref]
        metadata = images_module.build_picture_metadata(
            picture, doc, image, picture_id, doc_id, section_path_by_ref.get(picture.self_ref, [])
        )
        images_module.write_picture_outputs(image, metadata, images_dir)
        picture_ids.append(picture_id)

    doc_metadata = _build_document_metadata(
        doc_id, pdf_path, doc.num_pages(), stem, table_ids, picture_ids
    )
    _write_document_metadata(metadata_dir, stem, doc_metadata)


def process_pdf_md_only(converter: DocumentConverter, pdf_path: Path, output_dir: Path) -> None:
    doc = convert_pdf(converter, pdf_path)
    stem = pdf_path.stem
    doc_id = make_id(doc.origin.binary_hash, "doc", [])

    metadata_dir = _prepare_md_only_output_dirs(output_dir)

    markdown = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    (output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")

    doc_metadata = _build_document_metadata(doc_id, pdf_path, doc.num_pages(), stem, [], [])
    _write_document_metadata(metadata_dir, stem, doc_metadata)
