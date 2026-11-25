
from ingestion.chunker import chunk_pages


def test_chunker():
    pages = [
        {
            "document_name": "sample.pdf",
            "page_number": 1,
            "text": "A" * 2000
        }
    ]

    chunks = chunk_pages(pages, chunk_size=800, overlap=150)

    assert len(chunks) == 3

    for chunk in chunks:
        assert len(chunk["text"]) <= 800
        assert chunk["document_name"] == "sample.pdf"
        assert chunk["page_number"] == 1

    assert chunks[0]["text"][-150:] == chunks[1]["text"][:150]

    print(f"Chunking successful! Created {len(chunks)} chunks.")


if __name__ == "__main__":
    test_chunker()
