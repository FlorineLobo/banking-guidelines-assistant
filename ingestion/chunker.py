
def chunk_pages(pages, chunk_size=800, overlap=150):
    """
    Split PDF pages into overlapping chunks,
    preferring natural text boundaries.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap value.")

    all_chunks = []

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            if end < len(text):

                # Look for a natural boundary in the latter
                # half of the proposed chunk.
                search_start = start + chunk_size // 2

                candidates = [
                    text.rfind("\n\n", search_start, end),
                    text.rfind("\n", search_start, end),
                    text.rfind(". ", search_start, end),
                    text.rfind("; ", search_start, end)
                ]

                boundary = max(candidates)

                if boundary != -1:
                    end = boundary + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                all_chunks.append({
                    "chunk_id": len(all_chunks),
                    "document_name": page["document_name"],
                    "page_number": page["page_number"],
                    "text": chunk_text
                })

            if end >= len(text):
                break

            # Retain overlap for contextual continuity.
            next_start = max(start + 1, end - overlap)

            start = next_start

    return all_chunks
