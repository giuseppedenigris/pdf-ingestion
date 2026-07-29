import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter

from src.conversion import build_converter
from src.manifest import append_manifest_entry, read_successful_source_files
from src.pipeline import process_pdf_default, process_pdf_md_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown/HTML/PNG/JSON using Docling."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--md-only", action="store_true")
    return parser.parse_args()


def discover_pdfs(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.pdf"))


def process_single_pdf(
    converter: DocumentConverter, pdf_path: Path, output_dir: Path, md_only: bool
) -> None:
    if md_only:
        process_pdf_md_only(converter, pdf_path, output_dir)
    else:
        process_pdf_default(converter, pdf_path, output_dir)


def run(input_dir: Path, output_dir: Path, md_only: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    already_processed = read_successful_source_files(output_dir)
    pending = [p for p in discover_pdfs(input_dir) if p.name not in already_processed]

    converter = build_converter()

    # Processed sequentially by default: the pypdfium2 backend's memory footprint
    # makes concurrent conversions risky. Each iteration is self-contained (its
    # own try/except and manifest write), so switching to a process pool later
    # only requires swapping this loop for a map over process_single_pdf.
    for index, pdf_path in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] processing {pdf_path.name}")
        try:
            process_single_pdf(converter, pdf_path, output_dir, md_only)
        except Exception as exc:
            print(f"  failed: {exc}")
            append_manifest_entry(output_dir, pdf_path.name, "failed", str(exc))
        else:
            append_manifest_entry(output_dir, pdf_path.name, "success")


def main() -> None:
    args = parse_args()
    run(args.input_dir, args.output_dir, args.md_only)
