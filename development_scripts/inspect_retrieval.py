
from rag.knowledge_base_manager import KnowledgeBaseManager
from rag.retriever import Retriever


def main():
    manager = KnowledgeBaseManager()

    vector_store, embedding_model, rebuilt = manager.get_or_build()

    print("Knowledge base rebuilt:", rebuilt)
    print("Total chunks:", len(vector_store.chunks))

    retriever = Retriever(vector_store, embedding_model)

    questions = [
        "What is my degree and CGPA?",
        "What is the effective date of RBI guidelines?",
        "Compare the implementation dates of RBI and Jordan.",
        "What is the minimum CAR specified by RBI?"
    ]

    for question in questions:
        print("\n" + "=" * 90)
        print("QUESTION:", question)
        print("=" * 90)

        detected_documents = retriever._identify_relevant_documents(question)
        print("DETECTED DOCUMENTS:", detected_documents)

        results = retriever.retrieve(question, top_k=5)

        if not results:
            print("No results found.")
            continue

        for rank, chunk in enumerate(results, start=1):
            print(f"\nRESULT {rank}")
            print("-" * 60)

            print("Document:", chunk["document_name"])
            print("Page:", chunk["page_number"])
            print("Semantic similarity:", chunk.get("semantic_similarity"))
            print("Semantic rank:", chunk.get("semantic_rank"))
            print("Keyword overlap:", chunk.get("keyword_overlap"))
            print("Keyword rank:", chunk.get("keyword_rank"))
            print("RRF score:", chunk.get("rrf_score"))
            print("Relevance boost:", chunk.get("relevance_boost"))
            print("Hybrid score:", chunk.get("hybrid_score"))

            print("\nText preview:")
            print(chunk["text"][:350])


if __name__ == "__main__":
    main()
