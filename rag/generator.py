
import os
import re

from dotenv import load_dotenv
from google import genai

from rag.prompts import SYSTEM_PROMPT, build_user_prompt


load_dotenv()


INSUFFICIENT_INFORMATION = (
    "I could not find sufficient information to answer "
    "this question in the selected document(s)."
)


class AnswerGenerator:

    def __init__(self, model_name=None):

        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in .env file."
            )

        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or "gemini-3.5-flash-lite"
        )

        self.client = genai.Client(api_key=self.api_key)

    def validate_citations(self, answer, retrieved_chunks):
        """
        Check whether every cited document and page
        exists in the retrieved passages.
        """

        valid_sources = {
            (
                chunk["document_name"],
                str(chunk["page_number"])
            )
            for chunk in retrieved_chunks
        }

        # Extract citations in the format:
        # Document: filename.pdf
        # Page: 4
        citation_pattern = (
    r"Document:\s*(.+?)\s*\n\s*-?\s*Page:\s*(\d+)"
)
        citations = re.findall(
            citation_pattern,
            answer,
            flags=re.IGNORECASE
        )

        # If no citations exist, reject a purported answer.
        if not citations:
            return False

        for document_name, page_number in citations:

            citation = (
                document_name.strip(),
                page_number.strip()
            )

            if citation not in valid_sources:
                return False

        return True

    def generate_answer(self, question, retrieved_chunks):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not retrieved_chunks:
            return INSUFFICIENT_INFORMATION

        user_prompt = build_user_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        try:

            interaction = self.client.interactions.create(
                model=self.model_name,
                input=SYSTEM_PROMPT + "\n\n" + user_prompt
            )

            answer = interaction.output_text

            if not answer or not answer.strip():
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            answer = answer.strip()

            # Allow a clear abstention without citations.
            if INSUFFICIENT_INFORMATION.lower() in answer.lower():
                return INSUFFICIENT_INFORMATION

            # Validate citations before returning the answer.
            if not self.validate_citations(
                answer,
                retrieved_chunks
            ):
                return (
                    "The generated answer contained an invalid "
                    "or unverifiable citation. Please try again "
                    "or review the retrieved document passages."
                )

            return answer

        except Exception as error:

            raise RuntimeError(
                f"Failed to generate answer using Gemini: {error}"
            ) from error

    def close(self):
        self.client.close()
