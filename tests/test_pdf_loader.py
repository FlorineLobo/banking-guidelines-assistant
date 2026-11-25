
import pymupdf as fitz

import pytest

from ingestion.pdf_loader import load_pdf


def test_pdf_loader(tmp_path):
    """
    Test PDF text extraction using a temporary PDF.
    """

    # STEP 1: Create a temporary PDF.
    pdf_path = tmp_path / "test.pdf"

    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "This is a test banking document."
        )
        pdf.save(pdf_path)

    # STEP 2: Load the PDF.
    pages = load_pdf(pdf_path)

    # STEP 3: Verify extracted information.
    assert len(pages) == 1

    assert pages[0]["document_name"] == "test.pdf"

    assert pages[0]["page_number"] == 1

    assert "test banking document" in pages[0]["text"]


def test_pdf_loader_missing_file(tmp_path):

    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf(pdf_path)


def test_pdf_loader_invalid_extension(tmp_path):

    file_path = tmp_path / "document.txt"
    file_path.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        load_pdf(file_path)


def test_pdf_loader_empty_pdf(tmp_path):

    pdf_path = tmp_path / "empty.pdf"

    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(pdf_path)

    with pytest.raises(ValueError, match="No extractable text"):
        load_pdf(pdf_path)
