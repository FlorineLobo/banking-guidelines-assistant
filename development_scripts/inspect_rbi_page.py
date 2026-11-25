
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages


# Load only the RBI document
pdf_path = "data/uploads/ECL India Guidelines.pdf"

pages = load_pdf(pdf_path)

# Find page 4
page_four = [
    page for page in pages
    if page["page_number"] == 4
]

if not page_four:
    print("Page 4 was not extracted!")
else:
    print("\nEXTRACTED TEXT FROM RBI PAGE 4")
    print("=" * 80)
    print(page_four[0]["text"])


# Chunk page 4
chunks = chunk_pages(page_four, chunk_size=800, overlap=150)

print("\n\nCHUNKS CREATED FROM PAGE 4")
print("=" * 80)

for chunk in chunks:
    print("\nChunk ID:", chunk["chunk_id"])
    print("Page:", chunk["page_number"])
    print("Text:", chunk["text"])
    print("-" * 80)


# Check for commencement wording
print("\nCHECKING FOR COMMENCEMENT WORDING")

for chunk in chunks:
    if "force" in chunk["text"].lower():
        print("\nFound a matching chunk:")
        print(chunk["text"])
