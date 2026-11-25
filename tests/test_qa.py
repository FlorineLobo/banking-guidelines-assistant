
from build_knowledge_base import build_knowledge_base
from rag.retriever import Retriever
from rag.generator import AnswerGenerator


def test_rag_answers_and_handles_missing_information():

    print("\nBUILDING KNOWLEDGE BASE...")

    vector_store, embedding_model = build_knowledge_base()

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    generator = AnswerGenerator()

    test_cases = [
        {
            "question": (
                "What is the projected global speculative-grade "
                "default rate by the end of 2026?"
            ),
            "document": (
                "Default_Report-Annual-default-study-12Mar2026.pdf"
            ),
            "expected_page": 1,
            "expected_keyword": "3%",
            "should_abstain": False
        },
        {
            "question": (
                "When do the RBI Directions come into effect?"
            ),
            "document": "ECL India Guidelines.pdf",
            "expected_page": 4,
            "expected_keyword": "April 1, 2027",
            "should_abstain": False
        },
        {
            "question": (
                "What is the minimum regulatory capital ratio "
                "under the Jordan Basel III instructions?"
            ),
            "document": (
                "CBJ Regulatory Capital Insturctions Basel III No 67-2016.pdf"
            ),
            "expected_page": 19,
            "expected_keyword": "12%",
            "should_abstain": False
        },
        {
            "question": (
                "What is the minimum Capital Adequacy Ratio "
                "specified in the RBI ECL Guidelines?"
            ),
            "document": "ECL India Guidelines.pdf",
            "should_abstain": True
        }
    ]

    try:
        for case in test_cases:

            print("\n" + "=" * 60)
            print("QUESTION:", case["question"])

            retrieved_chunks = retriever.retrieve(
                question=case["question"],
                top_k=5,
                document_name=case["document"]
            )

            # Check that retrieval returned evidence
            assert retrieved_chunks, (
                f"No passages retrieved for: {case['question']}"
            )

            # Check document isolation
            assert all(
                chunk["document_name"] == case["document"]
                for chunk in retrieved_chunks
            ), "A passage from an unrelated document was retrieved."

            print("\nRETRIEVED SOURCES:")

            for chunk in retrieved_chunks:
                print(
                    f"Page {chunk['page_number']} | "
                    f"Score: {chunk['hybrid_score']:.5f}"
                )

            # Verify the expected page for answerable questions
            if not case["should_abstain"]:

                matching_source = [
                    chunk for chunk in retrieved_chunks
                    if chunk["page_number"] == case["expected_page"]
                ]

                assert matching_source, (
                    f"Expected page {case['expected_page']} "
                    "was not retrieved."
                )

            # Generate the answer
            answer = generator.generate_answer(
                question=case["question"],
                retrieved_chunks=retrieved_chunks
            )

            print("\nGENERATED ANSWER:")
            print(answer)

            if case["should_abstain"]:

                # The assistant must not borrow the Jordan CAR value
                assert "12%" not in answer, (
                    "The assistant incorrectly used the Jordan "
                    "document's CAR requirement."
                )

                assert "could not find sufficient information" in (
                    answer.lower()
                ), "The assistant did not clearly abstain."

            else:

                assert (
                    case["expected_keyword"].lower() in answer.lower()
                ), (
                    f"Expected information "
                    f"'{case['expected_keyword']}' was missing."
                )

            print("\nTEST PASSED!")

    finally:
        generator.close()
