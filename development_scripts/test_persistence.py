
from build_knowledge_base import build_knowledge_base
from rag.vector_store import VectorStore


print("\nSTEP 1: BUILDING KNOWLEDGE BASE")

vector_store, embedding_model = build_knowledge_base()

original_count = vector_store.index.ntotal

print("\nOriginal vectors:", original_count)


print("\nSTEP 2: SAVING KNOWLEDGE BASE")

vector_store.save()


print("\nSTEP 3: LOADING KNOWLEDGE BASE")

loaded_store = VectorStore.load()


print("\nSTEP 4: VERIFYING DATA")

assert loaded_store.index.ntotal == original_count

assert len(loaded_store.chunks) == len(vector_store.chunks)

assert (
    loaded_store.chunks[0]["document_name"]
    == vector_store.chunks[0]["document_name"]
)

assert (
    loaded_store.chunks[0]["page_number"]
    == vector_store.chunks[0]["page_number"]
)

print("Vector count matches!")
print("Chunk metadata matches!")


print("\nALL PERSISTENCE TESTS PASSED!")
