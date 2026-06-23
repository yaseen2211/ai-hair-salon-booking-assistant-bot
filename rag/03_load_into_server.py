"""
Embed knowledge_base/*.md files and store them in a running Chroma server.

PREREQUISITE: chroma run --path ./rag/chroma_db --port 8000 already running.

Run: python rag/03_load_into_server.py
"""

import os
import glob
import chromadb
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document

load_dotenv()

KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "slotpilot_policies"


def load_documents() -> list[Document]:
    documents = []
    for filepath in glob.glob(f"{KNOWLEDGE_BASE_DIR}/*.md"):
        with open(filepath, "r") as f:
            content = f.read()
        documents.append(Document(
            page_content=content,
            metadata={"source": os.path.basename(filepath)},
        ))
    return documents


if __name__ == "__main__":
    print(f"Connecting to Chroma server at {CHROMA_HOST}:{CHROMA_PORT} ...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    print("Connected.\n")

    docs = load_documents()
    print(f"Loaded {len(docs)} source files: {[d.metadata['source'] for d in docs]}")

    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks\n")

    embeddings = VoyageAIEmbeddings(
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-3.5",
    )

    print("Embedding chunks and writing to the Chroma SERVER ...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )
    print("Done.\n")

    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()
    print(f"Collection '{COLLECTION_NAME}' on the server now has {count} chunks.")

    data = collection.get(include=["documents", "metadatas"])
    print("\nStored chunks (confirmed via server query):")
    for i in range(len(data["ids"])):
        print(f"  [{i}] source={data['metadatas'][i].get('source')} "
              f"| preview={data['documents'][i][:60]}...")