import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter

from src import captioning
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
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Skip VLM/LLM captioning. Only the hard degenerate-content pre-check runs "
        "(filtered_out/filtered_reason stay at their defaults for everything else).",
    )
    parser.add_argument("--captioner-url", type=str, help="Base URL of the Ollama server.")
    parser.add_argument("--captioner-model", type=str, help="Ollama model name to use for captioning.")
    args = parser.parse_args()
    if not args.verify_output and args.input_dir is None:
        parser.error("--input-dir is required unless --verify-output is set")
    if not args.verify_output and not args.no_caption and (not args.captioner_url or not args.captioner_model):
        parser.error("--captioner-url and --captioner-model are required unless --no-caption is set")
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


def run(
    input_dir: Path,
    output_dir: Path,
    md_only: bool,
    save_log: bool,
    no_caption: bool,
    captioner_url: str | None,
    captioner_model: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_warnings(output_dir, save_log)

    already_processed = read_successful_source_files(output_dir)
    pending = [p for p in discover_pdfs(input_dir) if p.name not in already_processed]

    # One worklist, one pass: each item is driven from wherever it currently
    # is (not yet converted, or converted but not fully captioned) to done.
    work_items: list[tuple[str, Path | None]] = [(p.stem, p) for p in pending]
    work_items += [(Path(name).stem, None) for name in already_processed]
    work_items.sort(key=lambda item: item[0])

    converter = build_converter()
    captioner = None if no_caption else captioning.build_captioner(captioner_url, captioner_model)
    state = captioning.CaptionRunState()

    work_items = [
        (stem, pdf_path)
        for stem, pdf_path in work_items
        if pdf_path is not None or captioning.needs_enrichment(output_dir, stem)
    ]

    for index, (stem, pdf_path) in enumerate(work_items, start=1):
        if pdf_path is not None:
            size_mb = pdf_path.stat().st_size / 1_000_000
            prefix = f"[{index}/{len(work_items)}] {pdf_path.name} ({size_mb:.1f} MB)"
            spinner = Spinner(f"{prefix}: parsing")
            spinner.start()
            try:
                process_single_pdf(converter, pdf_path, output_dir, md_only)
            except Exception as exc:
                spinner.stop(f"{prefix} failed after {spinner.elapsed:.1f}s: {exc}")
                append_manifest_entry(output_dir, pdf_path.name, "failed", str(exc))
                continue
            append_manifest_entry(output_dir, pdf_path.name, "success")
        else:
            prefix = f"[{index}/{len(work_items)}] {stem}"
            spinner = Spinner(prefix)
            spinner.start()

        try:
            captioning.enrich_document(
                output_dir,
                stem,
                captioner,
                state,
                on_progress=lambda msg, prefix=prefix: spinner.update(f"{prefix}: {msg}"),
            )
        except captioning.CaptionerUnavailable as exc:
            spinner.stop(f"{prefix}: captioner unavailable, disabling captioning for rest of run ({exc})")
            captioner = None
        else:
            spinner.stop(f"{prefix} done in {spinner.elapsed:.1f}s")


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
    run(
        args.input_dir,
        args.output_dir,
        args.md_only,
        args.save_log,
        args.no_caption,
        args.captioner_url,
        args.captioner_model,
    )
