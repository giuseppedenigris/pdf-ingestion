from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument


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


def convert_pdf(converter: DocumentConverter, pdf_path: Path) -> DoclingDocument:
    result = converter.convert(pdf_path)
    return result.document
