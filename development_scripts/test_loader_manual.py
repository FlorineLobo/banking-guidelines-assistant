
from ingestion.pdf_loader import load_pdf

from pathlib import Path

# Find all PDF files in our uploads folder
pdf_folder = Path("data/uploads")

pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found!")
else:
    for pdf_file in pdf_files:
        print("\n" + "=" * 70)
        print(f"DOCUMENT: {pdf_file.name}")
        print("=" * 70)

        try:
            pages = load_pdf(pdf_file)

            print(f"Pages with extracted text: {len(pages)}")

            # Display the first page's extracted text
            print("\nFIRST PAGE EXTRACTED TEXT:")
            print(pages[0]["text"][:500])

            print("\nSOURCE METADATA:")
            print(f"Document: {pages[0]['document_name']}")
            print(f"Page number: {pages[0]['page_number']}")

        except (ValueError, FileNotFoundError) as error:
            print(f"ERROR: {error}")
