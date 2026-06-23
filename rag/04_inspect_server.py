"""
Independent inspection script — connects fresh to the Chroma SERVER and
shows what's stored, including raw embedding vectors. Run this anytime
the server is up, completely separately from the script that loaded the
data, to prove the data genuinely lives on the server.

PREREQUISITE: Chroma server running (chroma run --path ./rag/chroma_db --port 8000)

Run: python rag/04_inspect_server.py
"""

import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "slotpilot_policies"

if __name__ == "__main__":
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    collections = client.list_collections()
    print(f"Collections on server: {[c.name for c in collections]}\n")

    collection = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}': {collection.count()} chunks\n")

    data = collection.get(include=["documents", "metadatas", "embeddings"])

    for i in range(len(data["ids"])):
        print(f"{'='*60}")
        print(f"ID: {data['ids'][i]}")
        print(f"Source: {data['metadatas'][i].get('source')}")
        print(f"Text:\n{data['documents'][i]}")
        print(f"\nEmbedding (first 8 of {len(data['embeddings'][i])} dims):")
        print(data["embeddings"][i][:8])
        print()