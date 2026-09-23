import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter

from src import captioning
from src.conversion import build_converter
from src.integrity import IntegrityReport, verify_output
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
        help="Only check --output-dir for missing/orphaned files against parsed documents; no conversion runs.",
    )
    parser.add_argument("--ollama-url", type=str, help="Base URL of the Ollama server.")
    parser.add_argument("--captioner-images", type=str, help="Ollama model name for image captioning.")
    parser.add_argument("--captioner-tables", type=str, help="Ollama model name for table captioning.")
    args = parser.parse_args()
    if not args.verify_output and args.input_dir is None:
        parser.error("--input-dir is required unless --verify-output is set")
    if not args.verify_output and (args.captioner_images or args.captioner_tables) and not args.ollama_url:
        parser.error("--ollama-url is required when --captioner-images or --captioner-tables is set")
    return args


def discover_pdfs(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def process_single_pdf(
    converter: DocumentConverter, pdf_path: Path, output_dir: Path, md_only: bool
) -> None:
    if md_only:
        process_pdf_md_only(converter, pdf_path, output_dir)
    else:
        process_pdf_default(converter, pdf_path, output_dir)


def _is_parsed(output_dir: Path, stem: str) -> bool:
    return (output_dir / "metadata" / f"{stem}.json").exists()


def _collect_pdfs_to_process(
    input_dir: Path,
    output_dir: Path,
    images_active: bool,
    tables_active: bool,
) -> list[tuple[Path, bool, bool, bool]]:
    work_items: list[tuple[Path, bool, bool, bool]] = []
    for pdf_path in discover_pdfs(input_dir):
        stem = pdf_path.stem
        needs_parse = not _is_parsed(output_dir, stem)
        needs_images = images_active and (needs_parse or captioning.needs_image_captioning(output_dir, stem))
        needs_tables = tables_active and (needs_parse or captioning.needs_table_captioning(output_dir, stem))
        if needs_parse or needs_images or needs_tables:
            work_items.append((pdf_path, needs_parse, needs_images, needs_tables))
    return work_items


def run(
    input_dir: Path,
    output_dir: Path,
    md_only: bool,
    save_log: bool,
    ollama_url: str | None,
    captioner_images: str | None,
    captioner_tables: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_warnings(output_dir, save_log)

    image_captioner = captioning.build_image_captioner(ollama_url, captioner_images) if captioner_images else None
    table_captioner = captioning.build_table_captioner(ollama_url, captioner_tables) if captioner_tables else None

    if image_captioner is not None:
        print(f"info: image captioning with model '{captioner_images}' via {ollama_url}")
    else:
        print("info: skipping image captioning: no captioner provided (--captioner-images)")
    if table_captioner is not None:
        print(f"info: table captioning with model '{captioner_tables}' via {ollama_url}")
    else:
        print("info: skipping table captioning: no captioner provided (--captioner-tables)")

    work_items = _collect_pdfs_to_process(
        input_dir, output_dir, image_captioner is not None, table_captioner is not None
    )

    converter = build_converter()
    image_state = captioning.CaptionRunState()
    table_state = captioning.CaptionRunState()

    for index, (pdf_path, needs_parse, needs_images, needs_tables) in enumerate(work_items, start=1):
        stem = pdf_path.stem
        size_mb = pdf_path.stat().st_size / 1_000_000
        prefix = f"[{index}/{len(work_items)}] {pdf_path.name} ({size_mb:.1f} MB)"
        spinner = Spinner(f"{prefix}: parsing" if needs_parse else prefix)
        spinner.start()

        if needs_parse:
            try:
                process_single_pdf(converter, pdf_path, output_dir, md_only)
            except Exception as exc:
                spinner.stop(f"{prefix} failed after {spinner.elapsed:.1f}s: {exc}")
                continue

        summary_parts = []

        if needs_images:
            try:
                total, skipped, filtered, captioned = captioning.caption_images(
                    output_dir,
                    stem,
                    image_captioner,
                    image_state,
                    on_progress=lambda msg, prefix=prefix: spinner.update(f"{prefix}: {msg}"),
                )
                summary_parts.append(f"{total} images: {skipped}/{filtered}/{captioned}")
            except captioning.CaptionerUnavailable as exc:
                spinner.stop(
                    f"{prefix}: image captioner unavailable, disabling image captioning for rest of run ({exc})"
                )
                image_captioner = None

        if needs_tables:
            try:
                total, skipped, filtered, captioned = captioning.caption_tables(
                    output_dir,
                    stem,
                    table_captioner,
                    table_state,
                    on_progress=lambda msg, prefix=prefix: spinner.update(f"{prefix}: {msg}"),
                )
                summary_parts.append(f"{total} tables: {skipped}/{filtered}/{captioned}")
            except captioning.CaptionerUnavailable as exc:
                spinner.stop(
                    f"{prefix}: table captioner unavailable, disabling table captioning for rest of run ({exc})"
                )
                table_captioner = None

        summary = f" - {' - '.join(summary_parts)} (skipped/filtered/captioned)" if summary_parts else ""
        spinner.stop(f"{prefix} done in {spinner.elapsed:.1f}s{summary}")


def print_integrity_report(report: IntegrityReport) -> None:
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
        args.ollama_url,
        args.captioner_images,
        args.captioner_tables,
    )
