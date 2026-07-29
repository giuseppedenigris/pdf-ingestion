from datetime import datetime, timezone
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import ImageRefMode

from src import images as images_module
from src import output
from src import tables as tables_module
from src.conversion import convert_pdf
from src.ids import make_id
from src.section_path import compute_section_paths
from src.serializers import render_markdown_with_anchors


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

    metadata_dir, tables_dir, images_dir = output.prepare_default_output_dirs(output_dir)

    markdown = render_markdown_with_anchors(doc, table_id_map, picture_id_map)
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
    output.write_document_metadata(metadata_dir, stem, doc_metadata)


def process_pdf_md_only(converter: DocumentConverter, pdf_path: Path, output_dir: Path) -> None:
    doc = convert_pdf(converter, pdf_path)
    stem = pdf_path.stem
    doc_id = make_id(doc.origin.binary_hash, "doc", [])

    metadata_dir = output.prepare_md_only_output_dirs(output_dir)

    markdown = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    (output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")

    doc_metadata = _build_document_metadata(doc_id, pdf_path, doc.num_pages(), stem, [], [])
    output.write_document_metadata(metadata_dir, stem, doc_metadata)
