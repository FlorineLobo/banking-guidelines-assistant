
from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"


def build_test_retriever():
    """
    Build a temporary in-memory knowledge base from all uploaded PDFs.

    This allows us to test cross-document retrieval without depending
    on the previously saved FAISS index.
    """

    all_chunks = []

    pdf_files = sorted(UPLOADS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {UPLOADS_DIR}"
        )

    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)

        chunks = chunk_pages(
            pages,
            chunk_size=800,
            overlap=150
        )

        all_chunks.extend(chunks)

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.generate_embeddings(texts)

    vector_store = VectorStore(embedding_dim=384)

    vector_store.add_documents(
        all_chunks,
        embeddings
    )

    return Retriever(vector_store, embedding_model)


def normalize_text(text):
    """Normalize text to make assertions case-insensitive."""
    return " ".join(text.lower().split())


def test_cross_document_date_retrieval():
    """
    Verify that the retriever finds the correct effective/implementation
    date evidence from both regulatory documents.

    Expected:
    - RBI: Page 4, commencement of Directions, April 1, 2027.
    - Jordan: Page 19, implementation date, 9/30/2016.
    """

    retriever = build_test_retriever()

    question = (
        "Compare the effective dates of the RBI and Jordan "
        "regulatory documents."
    )

    # Identify which documents the retriever considers relevant.
    matched_documents = retriever._identify_relevant_documents(question)

    print("\nMatched documents:")
    for document in matched_documents:
        print(f"- {document}")

    # Retrieve the final five passages.
    results = retriever.retrieve(
        question,
        top_k=5
    )

    print("\nFinal retrieved passages:")

    for result in results:
        print("\n----------------------------------------")
        print(f"Document: {result['document_name']}")
        print(f"Page: {result['page_number']}")
        print(f"Score: {result.get('hybrid_score')}")
        print(f"Text: {result['text'][:400]}")

    assert results, "No passages were retrieved."

    retrieved_documents = {
        result["document_name"]
        for result in results
    }

    rbi_document = "ECL India Guidelines.pdf"

    jordan_document = (
        "CBJ Regulatory Capital Insturctions Basel III No 67-2016.pdf"
    )

    # Both regulatory documents must be represented.
    assert rbi_document in retrieved_documents, (
        "RBI document was not included in the retrieved results."
    )

    assert jordan_document in retrieved_documents, (
        "Jordan document was not included in the retrieved results."
    )

    # --------------------------------------------------
    # Validate RBI commencement evidence on page 4.
    # --------------------------------------------------

    rbi_page_4_chunks = [
        result["text"]
        for result in results
        if (
            result["document_name"] == rbi_document
            and result["page_number"] == 4
        )
    ]

    assert rbi_page_4_chunks, (
        "RBI commencement evidence from page 4 "
        "was not retrieved."
    )

    rbi_page_4_text = normalize_text(
        " ".join(rbi_page_4_chunks)
    )

    print("\nRBI page 4 evidence:")
    print(rbi_page_4_text)

    assert (
        "come into force" in rbi_page_4_text
        or "commencement" in rbi_page_4_text
    ), (
        "RBI page 4 was retrieved, but the commencement "
        "language was not found."
    )

    assert "april 1, 2027" in rbi_page_4_text, (
        "RBI page 4 does not contain the expected "
        "commencement date: April 1, 2027."
    )

    # --------------------------------------------------
    # Validate Jordan implementation evidence on page 19.
    # --------------------------------------------------

    jordan_page_19_chunks = [
        result["text"]
        for result in results
        if (
            result["document_name"] == jordan_document
            and result["page_number"] == 19
        )
    ]

    assert jordan_page_19_chunks, (
        "Jordan implementation evidence from page 19 "
        "was not retrieved."
    )

    jordan_page_19_text = normalize_text(
        " ".join(jordan_page_19_chunks)
    )

    print("\nJordan page 19 evidence:")
    print(jordan_page_19_text)

    assert (
        "9/30/2016" in jordan_page_19_text
        or "september 30, 2016" in jordan_page_19_text
    ), (
        "Jordan page 19 does not contain the expected "
        "implementation date: September 30, 2016."
    )

    assert (
        "implementation" in jordan_page_19_text
        or "implemented" in jordan_page_19_text
    ), (
        "Jordan page 19 contains the date, but the "
        "implementation context was not retrieved."
    )


def test_cross_document_capital_retrieval():
    """
    Verify that a question comparing capital requirements
    retrieves evidence from both RBI and Jordan documents.
    """

    retriever = build_test_retriever()

    question = (
        "Compare the minimum capital requirements "
        "under the RBI and Jordan regulatory guidelines."
    )

    results = retriever.retrieve(
        question,
        top_k=5
    )

    print("\nCapital comparison retrieval results:")

    for result in results:
        print("\n----------------------------------------")
        print(f"Document: {result['document_name']}")
        print(f"Page: {result['page_number']}")
        print(f"Text: {result['text'][:300]}")

    assert results, "No passages were retrieved."

    retrieved_documents = {
        result["document_name"]
        for result in results
    }

    assert "ECL India Guidelines.pdf" in retrieved_documents, (
        "RBI document was not retrieved for the capital question."
    )

    assert (
        "CBJ Regulatory Capital Insturctions Basel III No 67-2016.pdf"
        in retrieved_documents
    ), (
        "Jordan document was not retrieved for the capital question."
    )
