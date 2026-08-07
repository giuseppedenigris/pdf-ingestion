def image_filter_reason(width: int, height: int) -> str | None:
    if width <= 0 or height <= 0:
        return "degenerate_dimensions"
    return None


def table_filter_reason(num_rows: int, num_cols: int) -> str | None:
    # Not <=1: a legitimate single-column table (e.g. a parts list) is not degenerate.
    if num_rows == 0 or num_cols == 0:
        return "empty_table"
    return None
