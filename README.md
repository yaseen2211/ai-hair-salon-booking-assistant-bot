# AI Booking Agent — Tidy Cuts Demo

A working AI booking assistant built with **Claude**, **LangChain**, and **RAG**. It books appointments, answers policy questions grounded in real documents, and knows when to hand off to a human — instead of guessing.

This demo uses a fictional salon ("Tidy Cuts") with mock data. The pattern is adaptable to any appointment-based business — clinics, studios, consultants, and similar.

**Live demo:** _add your Streamlit Cloud link here_

---

## What it does

- Checks availability and books appointments using real tool calls
- Answers policy questions (cancellation, deposits, services, hours) by retrieving real documents — never guessing
- Escalates to a human for pricing, refunds, or anything outside its scope
- Remembers context across a conversation
- Keeps each visitor's bookings private and isolated — nothing shared, nothing saved to disk

Built with Claude (`claude-sonnet-4-6`), LangChain, Chroma (embedded vector store), Voyage AI embeddings, and Streamlit for the web interface.

---

## Project structure
├── app.py                            # Streamlit web interface

├── knowledge_base/                   # Source policy & service docs

├── rag/

│   ├── retriever.py                  # RAG tools + vector store

│   └── chroma_db/                    # Persisted embeddings (committed)

└── langchain_experiments/

├── booking_agent_core.py         # Shared agent logic

└── 02_complete_booking_agent.py  # Terminal interface

`booking_agent_core.py` holds all the actual logic. Both `app.py` and the terminal script are thin wrappers around it — there's one place to fix or extend the agent.

---

## Setup

```bash
git clone https://github.com/yaseen2211/ai-hair-salon-booking-assistant-bot.git
cd ai-hair-salon-booking-assistant-bot
python3.12 -m venv venv      # requires Python ≤3.13
source venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file:
ANTHROPIC_API_KEY=your-key-here

VOYAGE_API_KEY=your-key-here

First run only — build the vector store:

```bash
python rag/retriever.py
```

---

## Running it

**Terminal:**
```bash
cd langchain_experiments
python 02_complete_booking_agent.py
```

**Web (Streamlit):**
```bash
streamlit run app.py
```

---

## Updating the knowledge base

Editing a file in `knowledge_base/` doesn't auto-update the vector store. After editing, rebuild:

```bash
rm -rf rag/chroma_db
python rag/retriever.py
```

If deploying, commit the rebuilt `rag/chroma_db/` along with your edits.

---

## Try asking it

- "What's available on Tuesday?"
- "Do I need a deposit for balayage?"
- "What's your cancellation policy?"
- "How much does a haircut cost?" — watch it escalate instead of guessing

---

## Deploying

Deploys directly to [Streamlit Community Cloud](https://share.streamlit.io):

1. Push to GitHub (include `rag/chroma_db/` — the app needs it)
2. Connect the repo, set the main file to `app.py`
3. Add `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` under Settings → Secrets

No separate database or server needed — the vector store runs embedded.

---

## Scope

This is a working prototype: tool-calling, RAG-grounded answers, conversation memory, and guardrails against hallucinated actions. It uses mock availability and stores nothing persistently by design.

A production build adds a real calendar integration, persistent storage, and monitoring — happy to scope that for your business.