
from pathlib import Path

from ingestion.document_manager import DocumentManager
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from build_knowledge_base import build_knowledge_base


class KnowledgeBaseManager:

    def __init__(self):
        self.document_manager = DocumentManager()

        self.index_path = Path("data/processed/faiss_index.bin")
        self.chunks_path = Path("data/processed/chunks.json")

    def get_or_build(self):
        """
        Load the existing knowledge base if documents are unchanged.
        Otherwise, build and save a fresh knowledge base.
        """

        current_manifest = self.document_manager.get_current_manifest()

        if not current_manifest:
            raise ValueError("No PDF documents found in data/uploads.")

        saved_manifest = self.document_manager.load_saved_manifest()

        documents_changed = (
            saved_manifest is None
            or current_manifest != saved_manifest
        )

        index_exists = (
            self.index_path.exists()
            and self.chunks_path.exists()
        )

        # CASE 1: Existing knowledge base can be reused.
        if not documents_changed and index_exists:

            print("\nExisting documents are unchanged.")
            print("Loading saved FAISS knowledge base...")

            vector_store = VectorStore.load()
            embedding_model = EmbeddingModel()

            print("Knowledge base loaded successfully!")

            return vector_store, embedding_model, False

        # CASE 2: Documents are new or have changed.
        print("\nNew or modified documents detected.")
        print("Building a fresh knowledge base...")

        vector_store, embedding_model = build_knowledge_base()

        # Save FAISS index and chunk metadata.
        vector_store.save()

        # Save manifest only after successful index creation.
        self.document_manager.save_manifest(current_manifest)

        print("\nKnowledge base and document manifest saved!")

        return vector_store, embedding_model, True
