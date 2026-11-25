
from rag.knowledge_base_manager import KnowledgeBaseManager
from rag.retriever import Retriever

manager = KnowledgeBaseManager()

vector_store, embedding_model, rebuilt = manager.get_or_build()

# Initialize the retriever once, before the loop.
retriever = Retriever(vector_store, embedding_model)

target_document = "ECL India Guidelines.pdf"

for chunk in vector_store.chunks:

    if (
        chunk["document_name"] == target_document
        and chunk["page_number"] in [1, 4]
    ):

        text = chunk["text"].lower()

        if (
            chunk["page_number"] == 4
            or "commencement" in text
            or "april 27, 2026" in text
        ):

            print("\n" + "=" * 80)
            print("PAGE:", chunk["page_number"])
            print("CHUNK ID:", chunk["chunk_id"])
            print("=" * 80)

            print(chunk["text"])

            print(
                "\nContains commencement language:",
                retriever._contains_commencement_language(
                    chunk["text"]
                )
            )

            print(
                "Contains date:",
                retriever._contains_date(
                    chunk["text"]
                )
            )

            print(
                "Relevance boost:",
                retriever._calculate_relevance_boost(
                    "Compare the implementation dates of RBI and Jordan.",
                    chunk["text"]
                )
            )
