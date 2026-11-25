
from rag.knowledge_base_manager import KnowledgeBaseManager


manager = KnowledgeBaseManager()

print("\nTESTING KNOWLEDGE BASE MANAGER")

vector_store, embedding_model, rebuilt = manager.get_or_build()

print("\nRESULTS")
print("Knowledge base rebuilt:", rebuilt)
print("Total vectors:", vector_store.index.ntotal)
print("Total chunks:", len(vector_store.chunks))

assert vector_store.index.ntotal == len(vector_store.chunks)
assert vector_store.index.ntotal > 0

print("\nKNOWLEDGE BASE MANAGER TEST PASSED!")
