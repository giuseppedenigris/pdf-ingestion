import hashlib

from docling_core.types.doc.document import DocItem


def make_id(binary_hash: int, item_type: str, self_refs: list[str]) -> str:
    payload = f"{binary_hash}:{item_type}:{','.join(sorted(self_refs))}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def page_numbers(item: DocItem) -> list[int]:
    return [prov.page_no for prov in item.prov]


def bounding_box(item: DocItem) -> dict[str, float] | None:
    if not item.prov:
        return None
    bbox = item.prov[0].bbox
    return {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b}
