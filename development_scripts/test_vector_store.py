
import numpy as np

from rag.vector_store import VectorStore


# Three illustrative normalized vectors
embeddings = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.8, 0.6, 0.0]
], dtype=np.float32)


# Original document chunks and metadata
chunks = [
    {
        "chunk_id": 0,
        "document_name": "banking.pdf",
        "page_number": 1,
        "text": "Banks must maintain regulatory capital."
    },
    {
        "chunk_id": 1,
        "document_name": "rbi.pdf",
        "page_number": 5,
        "text": "Loan repayment and asset classification rules."
    },
    {
        "chunk_id": 2,
        "document_name": "basel.pdf",
        "page_number": 7,
        "text": "Capital requirements are calculated using risk-weighted assets."
    }
]


# Initialize vector store
vector_store = VectorStore(embedding_dim=3)

# Add document vectors
vector_store.add_documents(chunks, embeddings)

print("Vectors stored:", vector_store.index.ntotal)

# Search using a query vector
query = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)

results = vector_store.search(query, top_k=2)

print("\nTOP SEARCH RESULTS")
print("-" * 50)

for result in results:
    print("Document:", result["document_name"])
    print("Page:", result["page_number"])
    print("Score:", round(result["similarity_score"], 3))
    print("Text:", result["text"])
    print("-" * 50)
