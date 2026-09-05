import json
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DoclingDocument, TableItem

from src.utils import bounding_box, make_id, page_numbers


def build_table_id_map(doc: DoclingDocument, binary_hash: int) -> dict[str, str]:
    return {
        table.self_ref: make_id(binary_hash, "table", [table.self_ref]) for table in doc.tables
    }


def build_table_metadata(
    table: TableItem,
    doc: DoclingDocument,
    table_id: str,
    doc_id: str,
    section_path: list[str],
) -> dict[str, Any]:
    caption = table.caption_text(doc) or None
    return {
        "id": table_id,
        "type": "table",
        "doc_id": doc_id,
        "self_ref": table.self_ref,
        "page_no": page_numbers(table),
        "bbox": bounding_box(table),
        "section_path": section_path,
        "num_rows": table.data.num_rows,
        "num_cols": table.data.num_cols,
        "caption": caption,
        "caption_source": "native" if caption else "none",
        "retrieval_text": None,
        "filtered_out": False,
        "filtered_reason": None,
        "content_file": f"{table_id}.html",
    }


def write_table_outputs(
    table: TableItem,
    doc: DoclingDocument,
    metadata: dict[str, Any],
    tables_dir: Path,
) -> None:
    table_id = metadata["id"]
    # HTML instead of Markdown: Markdown tables cannot represent rowspan/colspan,
    # which docling's table structure model routinely produces.
    html = table.export_to_html(doc=doc)
    (tables_dir / f"{table_id}.html").write_text(html, encoding="utf-8")
    (tables_dir / "metadata" / f"{table_id}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
