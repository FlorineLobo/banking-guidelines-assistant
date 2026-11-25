
from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages


pdf_folder = Path("data/uploads")

pdf_files = list(pdf_folder.glob("*.pdf"))

for pdf_file in pdf_files:

    print("\n" + "=" * 70)
    print(f"DOCUMENT: {pdf_file.name}")
    print("=" * 70)

    # Step 1: Extract text from the PDF
    pages = load_pdf(pdf_file)

    # Step 2: Split extracted text into chunks
    chunks = chunk_pages(
        pages,
        chunk_size=800,
        overlap=150
    )

    print(f"Pages extracted: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")

    # Inspect the first chunk
    if chunks:
        first_chunk = chunks[0]

        print("\nFIRST CHUNK:")
        print(f"Chunk ID: {first_chunk['chunk_id']}")
        print(f"Document: {first_chunk['document_name']}")
        print(f"Page: {first_chunk['page_number']}")
        print(f"Characters: {len(first_chunk['text'])}")
        print(f"Text preview: {first_chunk['text'][:250]}")

print("\nAll documents processed!")
