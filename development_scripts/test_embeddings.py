
import numpy as np

from rag.embeddings import EmbeddingModel


# Step 1: Initialize the embedding model
embedding_model = EmbeddingModel()

# Step 2: Prepare sample sentences
sentences = [
    "Banks must maintain minimum regulatory capital.",
    "Financial institutions need to maintain adequate capital.",
    "The weather is sunny today."
]

# Step 3: Generate embeddings
embeddings = embedding_model.generate_embeddings(sentences)

# Step 4: Inspect the generated vectors
print("\nEMBEDDING RESULTS")
print("-" * 50)

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)
print("Dimensions per sentence:", embeddings.shape[1])

# Display a few values from the first vector
print("\nFirst 5 values of the first embedding:")
print(embeddings[0][:5])

# Step 5: Compare semantic similarity
similarity_matrix = np.dot(embeddings, embeddings.T)

print("\nSIMILARITY MATRIX")
print("-" * 50)
print(np.round(similarity_matrix, 3))

print("\nSIMILARITY INTERPRETATION")
print("Sentence 1 vs Sentence 2:",
      round(float(similarity_matrix[0][1]), 3))

print("Sentence 1 vs Sentence 3:",
      round(float(similarity_matrix[0][2]), 3))
