import pymupdf as fitz
from pathlib import Path


def load_pdf(file_path):
    """
    Extract text from every page of a PDF.

    Args:
        file_path: Path to the PDF document.

    Returns:
        A list of dictionaries containing document name,
        page number, and extracted text.
    """

    file_path = Path(file_path)

    # Check whether the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Check whether it is actually a PDF
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("The uploaded file must be a PDF.")

    extracted_pages = []

    try:
        with fitz.open(file_path) as pdf:

            # Check whether the PDF is encrypted
            if pdf.is_encrypted:
                raise ValueError(
                    f"The PDF is password-protected: {file_path.name}"
                )

            # Read every page
            for page_number, page in enumerate(pdf, start=1):

                text = page.get_text("text").strip()

                # Skip pages without extractable text
                if not text:
                    continue

                extracted_pages.append({
                    "document_name": file_path.name,
                    "page_number": page_number,
                    "text": text
                })

    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError(
            f"Could not read PDF '{file_path.name}': {error}"
        ) from error

    # Check whether any text was extracted
    if not extracted_pages:
        raise ValueError(
            f"No extractable text found in PDF: {file_path.name}"
        )

    return extracted_pages
