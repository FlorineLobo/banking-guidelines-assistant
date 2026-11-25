
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load the sentence-transformer embedding model.
        """

        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        """
        Convert a list of text strings into numerical vectors.
        """

        if not texts:
            raise ValueError("Cannot generate embeddings for empty text.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def generate_query_embedding(self, query):
        """
        Convert a user's question into a numerical vector.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding
