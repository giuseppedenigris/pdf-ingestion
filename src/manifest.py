import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_FILENAME = "_manifest.jsonl"


def manifest_path(output_dir: Path) -> Path:
    return output_dir / MANIFEST_FILENAME


def read_successful_source_files(output_dir: Path) -> set[str]:
    path = manifest_path(output_dir)
    if not path.exists():
        return set()

    successful: set[str] = set()
    with path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("status") == "success":
                successful.add(entry["source_file"])
    return successful


def append_manifest_entry(
    output_dir: Path,
    source_file: str,
    status: str,
    error: str | None = None,
) -> None:
    entry = {
        "source_file": source_file,
        "status": status,
        "error": error,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    with manifest_path(output_dir).open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
