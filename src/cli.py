import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter

from src.conversion import build_converter
from src.integrity import IntegrityReport, verify_output
from src.manifest import append_manifest_entry, read_successful_source_files
from src.logging import configure_warnings
from src.pipeline import process_pdf_default, process_pdf_md_only
from src.progress import Spinner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown/HTML/PNG/JSON using Docling."
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--md-only", action="store_true")
    parser.add_argument("--save-log", action="store_true")
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help="Only check --output-dir for missing/orphaned files against _manifest.jsonl; no conversion runs.",
    )
    args = parser.parse_args()
    if not args.verify_output and args.input_dir is None:
        parser.error("--input-dir is required unless --verify-output is set")
    return args


def discover_pdfs(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.pdf"))


def process_single_pdf(
    converter: DocumentConverter, pdf_path: Path, output_dir: Path, md_only: bool
) -> None:
    if md_only:
        process_pdf_md_only(converter, pdf_path, output_dir)
    else:
        process_pdf_default(converter, pdf_path, output_dir)


def run(input_dir: Path, output_dir: Path, md_only: bool, save_log: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_warnings(output_dir, save_log)
    already_processed = read_successful_source_files(output_dir)
    pending = [p for p in discover_pdfs(input_dir) if p.name not in already_processed]

    converter = build_converter()

    # Processed sequentially by default: the pypdfium2 backend's memory footprint
    # makes concurrent conversions risky. Each iteration is self-contained (its
    # own try/except and manifest write), so switching to a process pool later
    # only requires swapping this loop for a map over process_single_pdf.
    for index, pdf_path in enumerate(pending, start=1):
        size_mb = pdf_path.stat().st_size / 1_000_000
        label = f"[{index}/{len(pending)}] {pdf_path.name} ({size_mb:.1f} MB)"
        spinner = Spinner(label)
        spinner.start()
        try:
            process_single_pdf(converter, pdf_path, output_dir, md_only)
        except Exception as exc:
            spinner.stop(f"{label} failed after {spinner.elapsed:.1f}s: {exc}")
            append_manifest_entry(output_dir, pdf_path.name, "failed", str(exc))
        else:
            spinner.stop(f"{label} done in {spinner.elapsed:.1f}s")
            append_manifest_entry(output_dir, pdf_path.name, "success")


def print_integrity_report(report: IntegrityReport) -> None:
    if report.duplicate_successes:
        print(f"Duplicate manifest entries ({len(report.duplicate_successes)}):")
        for line in report.duplicate_successes:
            print(f"  {line}")
        print()

    if report.missing_files:
        print(f"Missing files ({len(report.missing_files)}):")
        for line in report.missing_files:
            print(f"  {line}")
        print()

    if report.orphan_files:
        print(f"Orphan files ({len(report.orphan_files)}):")
        for line in report.orphan_files:
            print(f"  {line}")
        print()

    print("--- Summary ---")
    print(f"Duplicate manifest entries: {len(report.duplicate_successes)}")
    print(f"Missing files: {len(report.missing_files)}")
    print(f"Orphan files: {len(report.orphan_files)}")
    print("All clean." if report.is_clean else "Problems found.")


def main() -> None:
    args = parse_args()
    if args.verify_output:
        report = verify_output(args.output_dir)
        print_integrity_report(report)
        raise SystemExit(0 if report.is_clean else 1)
    run(args.input_dir, args.output_dir, args.md_only, args.save_log)
