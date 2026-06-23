"""
RAG: retrieval tools, both general and topic-specific.

EMBEDDED MODE — Chroma runs in-process, persisting to a local folder
(rag/chroma_db). No separate server, no port, nothing external to run.
This is the deployment-friendly mode: the vector data folder gets
committed alongside your code and works wherever the code runs.

On first run, this will read knowledge_base/*.md, embed it via Voyage,
and build the local store. On subsequent runs, it loads the existing
folder instead of re-embedding.
"""

import os
import glob
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool

load_dotenv()

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge_base")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

_vector_store = None


def _load_documents() -> list[Document]:
    documents = []
    for filepath in glob.glob(f"{KNOWLEDGE_BASE_DIR}/*.md"):
        with open(filepath, "r") as f:
            content = f.read()
        documents.append(Document(
            page_content=content,
            metadata={"source": os.path.basename(filepath)},
        ))
    return documents


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embeddings = VoyageAIEmbeddings(
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-3.5",
    )

    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        _vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
    else:
        documents = _load_documents()
        splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        _vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    return _vector_store


@tool
def policy_lookup(question: str) -> str:
    """General-purpose lookup across ALL Tidy Cuts knowledge (policies,
    services, hours, location). Use this as a FALLBACK only when a
    question doesn't clearly belong to one specific topic, or spans
    multiple topics. For single-topic questions, prefer the more specific
    tool: lookup_cancellation_policy, lookup_deposit_policy,
    lookup_services, or lookup_hours_and_location."""
    vector_store = get_vector_store()
    results = vector_store.similarity_search(question, k=2)

    if not results:
        return "No relevant policy information found for this question."

    return "\n\n".join(
        f"[Source: {r.metadata['source']}]\n{r.page_content}"
        for r in results
    )


def _lookup_by_source(question: str, source_filename: str) -> str:
    vector_store = get_vector_store()
    results = vector_store.similarity_search(
        question,
        k=3,
        filter={"source": source_filename},
    )
    if not results:
        return f"No information found in {source_filename} for this question."
    return "\n\n".join(r.page_content for r in results)


@tool
def lookup_cancellation_policy(question: str) -> str:
    """Look up Tidy Cuts' cancellation policy specifically — late
    cancellation fees, no-show charges, and notice requirements."""
    return _lookup_by_source(question, "cancellation_policy.md")


@tool
def lookup_deposit_policy(question: str) -> str:
    """Look up Tidy Cuts' deposit policy specifically — which services
    require a deposit, the deposit amount, and refund rules."""
    return _lookup_by_source(question, "deposit_policy.md")


@tool
def lookup_services(question: str) -> str:
    """Look up Tidy Cuts' service descriptions specifically — what each
    service includes, typical duration, and prerequisites."""
    return _lookup_by_source(question, "services.md")


@tool
def lookup_hours_and_location(question: str) -> str:
    """Look up Tidy Cuts' hours and location specifically — open days,
    times, address, and parking."""
    return _lookup_by_source(question, "hours_location.md")


if __name__ == "__main__":
    print(lookup_deposit_policy.invoke({"question": "Do I need a deposit for balayage?"}))
    print(f"\n{'-'*60}\n")
    print(lookup_cancellation_policy.invoke({"question": "What if I miss my appointment?"}))