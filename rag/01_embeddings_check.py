"""
RAG Stage 1: Embeddings sanity check.

Confirms Voyage AI embeddings work in isolation, before any vector store
or LangChain retrieval is involved. We're just turning text into numbers
and looking at the shape — nothing else.

You'll need a Voyage AI API key: https://www.voyageai.com/
Add VOYAGE_API_KEY=your-key-here to your .env file.

Run: python rag/01_embeddings_check.py
"""

import os
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.environ["VOYAGE_API_KEY"],
    model="voyage-3.5",
)

if __name__ == "__main__":
    test_text = "Appointments can be cancelled up to 24 hours in advance."

    vector = embeddings.embed_query(test_text)

    print(f"Input text: {test_text!r}")
    print(f"Vector type: {type(vector)}")
    print(f"Vector length (dimensions): {len(vector)}")
    print(f"First 5 values: {vector[:5]}")

    similar_text = "You can cancel your booking a day before without a fee."
    different_text = "The salon is located at 123 Main Street."

    import numpy as np

    vec_similar = embeddings.embed_query(similar_text)
    vec_different = embeddings.embed_query(different_text)

    def cosine_similarity(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"\nSimilarity to a related sentence about cancellation: "
          f"{cosine_similarity(vector, vec_similar):.4f}")
    print(f"Similarity to an unrelated sentence about location: "
          f"{cosine_similarity(vector, vec_different):.4f}")
    print("\n(Expect the first number meaningfully higher than the second "
          "— that's the whole basis retrieval relies on.)")