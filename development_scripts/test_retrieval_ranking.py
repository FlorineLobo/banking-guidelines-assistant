
from rag.knowledge_base_manager import KnowledgeBaseManager
from rag.retriever import Retriever

manager = KnowledgeBaseManager()

vector_store, embedding_model, rebuilt = (
    manager.get_or_build()
)

retriever = Retriever(
    embedding_model=embedding_model,
    vector_store=vector_store
)

question = (
    "What is the implementation date of the "
    "Jordan Basel III capital requirements?"
)

results = retriever._search_document(
    question=question,
    document_name=(
        "CBJ Regulatory Capital Insturctions "
        "Basel III No 67-2016.pdf"
    ),
    top_k=10
)

print("\nJORDAN DOCUMENT — TOP 10 RESULTS")

for index, result in enumerate(results, start=1):

    print(f"\nRANK: {index}")
    print("Page:", result["page_number"])
    print("Semantic score:", result["similarity_score"])
    print("Keyword score:", result["keyword_score"])
    print("Hybrid score:", result["hybrid_score"])
    print("Text:", result["text"][:350])

print("\nDIAGNOSTIC COMPLETE")
