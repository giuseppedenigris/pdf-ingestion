from docling_core.types.doc.document import DoclingDocument, SectionHeaderItem, TitleItem


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
