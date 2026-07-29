from typing import Any

from docling_core.transforms.serializer.base import BaseTableSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownPictureSerializer,
)
from docling_core.types.doc.document import DoclingDocument, PictureItem, TableItem


class AnchorTableSerializer(BaseTableSerializer):
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


class AnchorPictureSerializer(MarkdownPictureSerializer):
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


def render_markdown_with_anchors(
    doc: DoclingDocument,
    table_id_map: dict[str, str],
    picture_id_map: dict[str, str],
) -> str:
    serializer = MarkdownDocSerializer(
        doc=doc,
        table_serializer=AnchorTableSerializer(table_id_map),
        picture_serializer=AnchorPictureSerializer(picture_id_map),
    )
    return serializer.serialize().text
