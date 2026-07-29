import hashlib


def make_id(binary_hash: int, item_type: str, self_refs: list[str]) -> str:
    payload = f"{binary_hash}:{item_type}:{','.join(sorted(self_refs))}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
