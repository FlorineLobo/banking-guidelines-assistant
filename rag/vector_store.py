
import faiss
import numpy as np
import json

from pathlib import Path


class VectorStore:

    def __init__(self, embedding_dim=384):

        self.embedding_dim = embedding_dim

        # Create FAISS inner-product index
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Store original document chunks
        self.chunks = []

    def add_documents(self, chunks, embeddings):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if len(chunks) == 0:
            raise ValueError("No document chunks provided.")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim} dimensions, "
                f"received {embeddings.shape[1]}."
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain invalid numerical values."
            )

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k=5):

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (1, self.embedding_dim):
            raise ValueError(
                f"Query embedding must have shape "
                f"(1, {self.embedding_dim})."
            )

        if not np.isfinite(query_embedding).all():
            raise ValueError(
                "Query embedding contains invalid numerical values."
            )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index_position in zip(scores[0], indices[0]):

            if index_position == -1:
                continue

            chunk = self.chunks[index_position]

            results.append({
                **chunk,
                "similarity_score": float(score)
            })

        return results

    def save(self, directory="data/processed"):
        """
        Save the FAISS index and chunk metadata to disk.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "faiss_index.bin"
        chunks_path = directory / "chunks.json"

        # Save numerical vectors
        faiss.write_index(
            self.index,
            str(index_path)
        )

        # Save original text and metadata
        with open(chunks_path, "w", encoding="utf-8") as file:
            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

        print("\nKNOWLEDGE BASE SAVED!")
        print("FAISS index:", index_path)
        print("Chunk metadata:", chunks_path)

    @classmethod
    def load(cls, directory="data/processed"):
        """
        Load a previously saved FAISS index and its metadata.
        """

        directory = Path(directory)

        index_path = directory / "faiss_index.bin"
        chunks_path = directory / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                "Saved knowledge base files were not found."
            )

        # Load FAISS index
        index = faiss.read_index(str(index_path))

        # Load chunk metadata
        with open(chunks_path, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        if index.d != 384:
            raise ValueError(
                f"Unexpected embedding dimension: {index.d}"
            )

        if index.ntotal != len(chunks):
            raise ValueError(
                "FAISS vector count does not match metadata count."
            )

        vector_store = cls(embedding_dim=index.d)

        vector_store.index = index
        vector_store.chunks = chunks

        print("\nKNOWLEDGE BASE LOADED!")
        print("Vectors:", index.ntotal)
        print("Chunks:", len(chunks))

        return vector_store
