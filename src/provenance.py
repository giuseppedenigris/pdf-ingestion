from docling_core.types.doc.document import DocItem


def page_numbers(item: DocItem) -> list[int]:
    return [prov.page_no for prov in item.prov]


def bounding_box(item: DocItem) -> dict[str, float] | None:
    if not item.prov:
        return None
    bbox = item.prov[0].bbox
    return {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b}
