
from build_knowledge_base import build_knowledge_base
from rag.retriever import Retriever


# ============================================================
# STEP 1: BUILD THE KNOWLEDGE BASE
# ============================================================

print("\nBUILDING KNOWLEDGE BASE")
print("=" * 80)

vector_store, embedding_model = build_knowledge_base()


# ============================================================
# STEP 2: INITIALIZE THE RETRIEVER
# ============================================================

retriever = Retriever(
    embedding_model=embedding_model,
    vector_store=vector_store
)

print("\nRETRIEVER INITIALIZED SUCCESSFULLY!")


# ============================================================
# HELPER FUNCTION: DISPLAY RETRIEVAL RESULTS
# ============================================================

def display_results(results):

    if not results:
        print("\nNo passages retrieved.")
        return

    for rank, result in enumerate(results, start=1):

        print(f"\nRANK: {rank}")
        print("Document:", result["document_name"])
        print("Page:", result["page_number"])

        print(
            "Semantic Score:",
            round(result["similarity_score"], 4)
        )

        print(
            "Keyword Score:",
            round(result["keyword_score"], 4)
        )

        print(
            "Hybrid Score:",
            round(result["hybrid_score"], 6)
        )

        print("Text:", result["text"][:500])
        print("-" * 80)


# ============================================================
# STEP 3: SEARCH ACROSS ALL UPLOADED DOCUMENTS
# ============================================================

question = (
    "What is the minimum regulatory capital requirement "
    "under the Basel III instructions?"
)

results = retriever.retrieve(
    question=question,
    top_k=5
)

print("\n\nALL-DOCUMENT SEARCH")
print("=" * 80)
print("QUESTION:", question)

display_results(results)


# ============================================================
# STEP 4: SEARCH WITHIN ONE SPECIFIC DOCUMENT
# ============================================================

jordan_document = (
    "CBJ Regulatory Capital Insturctions Basel III No 67-2016.pdf"
)

filtered_results = retriever.retrieve(
    question=question,
    top_k=5,
    document_name=jordan_document
)

print("\n\nDOCUMENT-FILTERED SEARCH")
print("=" * 80)
print("QUESTION:", question)
print("SEARCHING ONLY:", jordan_document)

display_results(filtered_results)


# ============================================================
# STEP 5: TEST MULTIPLE QUESTIONS ACROSS ALL DOCUMENTS
# ============================================================

test_questions = [

    "What is the minimum regulatory capital requirement "
    "under the Basel III instructions?",

    "When do the RBI ECL guidelines come into force?",

    "What is the expected global speculative-grade default "
    "rate by the end of 2026?",

    "What is the minimum Capital Adequacy Ratio prescribed "
    "in the RBI guidelines?"
]

print("\n\nMULTIPLE QUESTION RETRIEVAL TEST")
print("=" * 80)

for question in test_questions:

    print("\nQUESTION:", question)
    print("-" * 80)

    results = retriever.retrieve(
        question=question,
        top_k=5
    )

    display_results(results)


# ============================================================
# STEP 6: TEST DOCUMENT-SPECIFIC RETRIEVAL
# ============================================================

test_cases = [

    {
        "question": "When do the RBI ECL guidelines come into force?",
        "document": "ECL India Guidelines.pdf"
    },

    {
        "question": "What is the minimum Capital Adequacy Ratio?",
        "document": "ECL India Guidelines.pdf"
    },

    {
        "question": "What is the minimum regulatory capital requirement?",
        "document": jordan_document
    }
]

print("\n\nDOCUMENT-SPECIFIC RETRIEVAL TEST")
print("=" * 80)

for test in test_cases:

    print("\nQUESTION:", test["question"])
    print("SEARCHING DOCUMENT:", test["document"])
    print("-" * 80)

    results = retriever.retrieve(
        question=test["question"],
        top_k=15,
        document_name=test["document"]
    )

    display_results(results)


# ============================================================
# STEP 7: RBI COMMENCEMENT DATE — RANKING ANALYSIS
# ============================================================

print("\n\nRBI COMMENCEMENT DATE — RETRIEVAL RANKING")
print("=" * 80)

question = "When do the RBI Directions come into effect?"

rbi_document = "ECL India Guidelines.pdf"

results = retriever.retrieve(
    question=question,
    top_k=100,
    document_name=rbi_document
)

target_found = False

for rank, result in enumerate(results, start=1):

    print(
        f"Rank: {rank} | "
        f"Page: {result['page_number']} | "
        f"Semantic: {result['similarity_score']:.4f} | "
        f"Keyword: {result['keyword_score']:.4f} | "
        f"Hybrid: {result['hybrid_score']:.6f}"
    )

    if "come into force" in result["text"].lower():

        target_found = True

        print("\n>>> FOUND THE COMMENCEMENT PASSAGE <<<")
        print("Document:", result["document_name"])
        print("Page:", result["page_number"])
        print("Passage:", result["text"])
        print("-" * 80)

if not target_found:
    print("\nCommencement passage not found in the top 100 results.")


# ============================================================
# STEP 8: VERIFY DOCUMENT ISOLATION
# ============================================================

print("\n\nRBI CAPITAL ADEQUACY — DOCUMENT ISOLATION TEST")
print("=" * 80)

question = (
    "What is the minimum Capital Adequacy Ratio "
    "prescribed in the RBI guidelines?"
)

results = retriever.retrieve(
    question=question,
    top_k=5,
    document_name=rbi_document
)

print("QUESTION:", question)
print("SEARCHING ONLY:", rbi_document)

display_results(results)

# Check whether the retrieved RBI passages mention
# a specific minimum Capital Adequacy Ratio.

car_terms = [
    "capital adequacy ratio",
    "minimum capital ratio",
    "minimum regulatory capital",
    "capital requirement"
]

matching_passages = []

for result in results:

    text = result["text"].lower()

    if any(term in text for term in car_terms):
        matching_passages.append(result)

print("\nRBI CAPITAL REQUIREMENT CHECK")
print("-" * 80)

if matching_passages:

    print(
        "Potentially relevant capital-related passages found."
    )

    print(
        "These passages must be examined before deciding "
        "whether the document supports an answer."
    )

else:

    print(
        "No capital-requirement passages found in the "
        "top 5 retrieved results."
    )

print(
    "\nIMPORTANT: Retrieval results alone do not prove "
    "that the document contains or does not contain an answer."
)


# ============================================================
# END OF RETRIEVAL TESTING
# ============================================================

print("\n" + "=" * 80)
print("ALL RETRIEVAL TESTS COMPLETED!")
print("=" * 80)
