
from rag.knowledge_base_manager import KnowledgeBaseManager
from rag.retriever import Retriever
from rag.generator import AnswerGenerator


manager = KnowledgeBaseManager()

vector_store, embedding_model, rebuilt = (
    manager.get_or_build()
)

retriever = Retriever(
    embedding_model=embedding_model,
    vector_store=vector_store
)

generator = AnswerGenerator()

question = (
    "Compare the effective date of the RBI Directions "
    "with the implementation date of the Jordan Basel III "
    "capital requirements."
)

print("\nQUESTION:")
print(question)

retrieved_chunks = retriever.retrieve(
    question=question,
    top_k=5
)

print("\nGENERATING ANSWER...\n")

answer = generator.generate_answer(
    question=question,
    retrieved_chunks=retrieved_chunks
)

print("\nFINAL ANSWER:")
print(answer)

generator.close()
