
from build_knowledge_base import build_knowledge_base
from rag.retriever import Retriever
from rag.generator import AnswerGenerator


def main():

    print("\nBUILDING KNOWLEDGE BASE...")
    vector_store, embedding_model = build_knowledge_base()

    print("\nINITIALIZING RETRIEVER...")
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    print("\nINITIALIZING GEMINI...")
    generator = AnswerGenerator()

    question = "When do the RBI Directions come into effect?"

    print("\nQUESTION:")
    print(question)

    print("\nRETRIEVING RELEVANT PASSAGES...")

    retrieved_chunks = retriever.retrieve(
        question=question,
        top_k=5,
        document_name="ECL India Guidelines.pdf"
    )

    print("\nRETRIEVED PASSAGES:")
    for chunk in retrieved_chunks:
        print(
            f"\nDocument: {chunk['document_name']}"
            f"\nPage: {chunk['page_number']}"
            f"\nSimilarity: {chunk['similarity_score']:.4f}"
            f"\nContent: {chunk['text'][:500]}"
        )

    print("\nGENERATING ANSWER WITH GEMINI...")

    try:
        answer = generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        print("\n" + "=" * 60)
        print("FINAL RAG ANSWER")
        print("=" * 60)
        print(answer)

    finally:
        generator.close()


if __name__ == "__main__":
    main()
