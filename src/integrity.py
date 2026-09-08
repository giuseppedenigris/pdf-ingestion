import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IntegrityReport:
    missing_files: list[str] = field(default_factory=list)
    orphan_files: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.missing_files or self.orphan_files)


def verify_output(output_dir: Path) -> IntegrityReport:
    metadata_dir = output_dir / "metadata"
    images_dir = output_dir / "images"
    tables_dir = output_dir / "tables"

    stems = [p.stem for p in metadata_dir.glob("*.json")]

    expected: set[str] = set()
    missing_files: list[str] = []

    for stem in stems:
        expected.add(f"{stem}.md")
        expected.add(f"metadata/{stem}.json")

        md_path = output_dir / f"{stem}.md"
        if not md_path.exists():
            missing_files.append(f"{stem}.md")

        meta_path = metadata_dir / f"{stem}.json"
        doc = json.loads(meta_path.read_text(encoding="utf-8"))

        for picture_id in doc.get("picture_ids", []):
            expected.add(f"images/{picture_id}.png")
            expected.add(f"images/metadata/{picture_id}.json")
            if not (images_dir / f"{picture_id}.png").exists():
                missing_files.append(f"images/{picture_id}.png")
            if not (images_dir / "metadata" / f"{picture_id}.json").exists():
                missing_files.append(f"images/metadata/{picture_id}.json")

        for table_id in doc.get("table_ids", []):
            expected.add(f"tables/{table_id}.html")
            expected.add(f"tables/metadata/{table_id}.json")
            if not (tables_dir / f"{table_id}.html").exists():
                missing_files.append(f"tables/{table_id}.html")
            if not (tables_dir / "metadata" / f"{table_id}.json").exists():
                missing_files.append(f"tables/metadata/{table_id}.json")

    actual: set[str] = set()
    actual.update(p.name for p in output_dir.glob("*.md"))
    actual.update(f"metadata/{p.name}" for p in metadata_dir.glob("*.json"))
    actual.update(f"images/{p.name}" for p in images_dir.glob("*.png"))
    actual.update(f"images/metadata/{p.name}" for p in (images_dir / "metadata").glob("*.json"))
    actual.update(f"tables/{p.name}" for p in tables_dir.glob("*.html"))
    actual.update(f"tables/metadata/{p.name}" for p in (tables_dir / "metadata").glob("*.json"))

    orphan_files = sorted(actual - expected)

    return IntegrityReport(missing_files=missing_files, orphan_files=orphan_files)
