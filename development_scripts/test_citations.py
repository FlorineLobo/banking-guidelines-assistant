
from rag.generator import AnswerGenerator


# Create an object without calling the API.
# We only need to test the validation function.
generator = AnswerGenerator.__new__(AnswerGenerator)


retrieved_chunks = [
    {
        "document_name": "ECL India Guidelines.pdf",
        "page_number": 4,
        "text": "These Directions shall come into force from April 1, 2027."
    },
    {
        "document_name": "ECL India Guidelines.pdf",
        "page_number": 58,
        "text": "RBI may issue necessary clarifications."
    }
]


# Test 1: Correct citation
correct_answer = """
Answer:
The Directions come into force on April 1, 2027.

Sources:
- Document: ECL India Guidelines.pdf
- Page: 4
"""

# Test 2: Invalid page
invalid_page_answer = """
Answer:
The Directions come into force on April 1, 2027.

Sources:
- Document: ECL India Guidelines.pdf
- Page: 99
"""

# Test 3: Invalid document
invalid_document_answer = """
Answer:
The Directions come into force on April 1, 2027.

Sources:
- Document: Jordan Guidelines.pdf
- Page: 4
"""


assert generator.validate_citations(
    correct_answer,
    retrieved_chunks
) is True

assert generator.validate_citations(
    invalid_page_answer,
    retrieved_chunks
) is False

assert generator.validate_citations(
    invalid_document_answer,
    retrieved_chunks
) is False


print("Correct citation test: PASSED")
print("Invalid page test: PASSED")
print("Invalid document test: PASSED")
print("\nALL CITATION TESTS PASSED!")
