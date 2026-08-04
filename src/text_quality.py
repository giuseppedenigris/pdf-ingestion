import re

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def has_control_chars(text: str) -> bool:
    return bool(CONTROL_CHAR_RE.search(text))
