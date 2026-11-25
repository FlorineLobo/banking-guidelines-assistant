
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
    "Compare the effective date of the RBI Directions "
    "with the implementation date of the Jordan Basel III "
    "capital requirements."
)

results = retriever.retrieve(
    question=question,
    top_k=5
)

print("\nRETRIEVED EVIDENCE")

for index, result in enumerate(results, start=1):

    print(f"\nRESULT {index}")
    print("Document:", result["document_name"])
    print("Page:", result["page_number"])
    print("Hybrid score:", result["hybrid_score"])
    print("Text:", result["text"])

rbi_results = [
    result for result in results
    if "ECL India Guidelines" in result["document_name"]
]

jordan_results = [
    result for result in results
    if "CBJ Regulatory Capital" in result["document_name"]
]

assert rbi_results, "RBI evidence missing!"

assert jordan_results, "Jordan evidence missing!"

# Check whether the retrieved text actually contains
# the dates needed to answer the comparison.

rbi_text = " ".join(
    result["text"] for result in rbi_results
).lower()

jordan_text = " ".join(
    result["text"] for result in jordan_results
).lower()

assert (
    "april 1, 2027" in rbi_text
    or "april 1 2027" in rbi_text
), "RBI effective-date evidence not retrieved!"

assert (
    "9/30/2016" in jordan_text
    or "september 30, 2016" in jordan_text
    or "september 30 2016" in jordan_text
), "Jordan implementation-date evidence not retrieved!"

print("\nCROSS-DOCUMENT EVIDENCE TEST PASSED!")
