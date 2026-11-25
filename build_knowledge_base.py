
from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


def build_knowledge_base():

    pdf_folder = Path("data/uploads")
    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF documents found in data/uploads.")

    all_pages = []

    # STEP 1: Extract text from every PDF
    print("\nSTEP 1: EXTRACTING PDF TEXT")

    for pdf_file in pdf_files:

        pages = load_pdf(pdf_file)
        all_pages.extend(pages)

        print(f"{pdf_file.name}: {len(pages)} pages")

    # STEP 2: Create text chunks
    print("\nSTEP 2: CREATING CHUNKS")

    all_chunks = chunk_pages(
        all_pages,
        chunk_size=800,
        overlap=150
    )

    print("Total chunks:", len(all_chunks))

    # STEP 3: Generate embeddings
    print("\nSTEP 3: GENERATING EMBEDDINGS")

    embedding_model = EmbeddingModel()

    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = embedding_model.generate_embeddings(texts)

    print("Embedding shape:", embeddings.shape)

    # STEP 4: Store embeddings in FAISS
    print("\nSTEP 4: BUILDING FAISS INDEX")

    vector_store = VectorStore(embedding_dim=384)

    vector_store.add_documents(
        all_chunks,
        embeddings
    )

    print("Vectors stored:", vector_store.index.ntotal)

    return vector_store, embedding_model


if __name__ == "__main__":

    vector_store, embedding_model = build_knowledge_base()

    print("\nKNOWLEDGE BASE BUILT SUCCESSFULLY!")
