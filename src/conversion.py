from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMode,
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

from src.text_quality import has_control_chars

_ocr_converter: DocumentConverter | None = None


def build_converter() -> DocumentConverter:
    # The default docling-parse backend leaks memory on PDFs with many pages and
    # eventually raises std::bad_alloc, which then poisons every later conversion
    # in the same process. pypdfium2 does not have this issue.
    #
    # generate_picture_images must be on, otherwise PictureItem.get_image(doc) has
    # no crop to return and comes back None for every picture in the document.
    pipeline_options = PdfPipelineOptions(generate_picture_images=True)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=PyPdfiumDocumentBackend, pipeline_options=pipeline_options
            )
        }
    )


def _get_ocr_converter() -> DocumentConverter:
    # Some PDFs embed a font with no ToUnicode table (old Windows print drivers),
    # so the programmatic text layer decodes to garbage control characters even
    # though the page renders correctly. Forcing full-page OCR reads the
    # rendered pixels instead, sidestepping the broken text layer. Built lazily
    # and memoized: only the small minority of PDFs that actually need this
    # retry should pay for loading the OCR model.
    global _ocr_converter
    if _ocr_converter is None:
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True,
            # backend="torch": the default "onnxruntime" isn't installed in this project.
            ocr_options=RapidOcrOptions(
                mode=OcrMode.FULL_PAGE, backend="torch", lang=["english"]
            ),
        )
        _ocr_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=PyPdfiumDocumentBackend, pipeline_options=pipeline_options
                )
            }
        )
    return _ocr_converter


def convert_pdf(converter: DocumentConverter, pdf_path: Path) -> DoclingDocument:
    doc = converter.convert(pdf_path).document
    if has_control_chars(doc.export_to_markdown()):
        doc = _get_ocr_converter().convert(pdf_path).document
    return doc
