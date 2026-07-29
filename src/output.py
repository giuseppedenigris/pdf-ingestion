import json
from pathlib import Path
from typing import Any


def prepare_default_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    metadata_dir = output_dir / "metadata"
    tables_dir = output_dir / "tables"
    images_dir = output_dir / "images"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (images_dir / "metadata").mkdir(parents=True, exist_ok=True)
    return metadata_dir, tables_dir, images_dir


def prepare_md_only_output_dirs(output_dir: Path) -> Path:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir


def write_document_metadata(metadata_dir: Path, stem: str, metadata: dict[str, Any]) -> None:
    (metadata_dir / f"{stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
