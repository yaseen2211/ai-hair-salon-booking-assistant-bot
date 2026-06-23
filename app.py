"""
Streamlit web chat interface for the Tidy Cuts Booking Assistant.

Each browser session gets its own isolated, in-memory bookings list —
nothing shared between visitors, nothing written to disk.

Run: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "langchain_experiments"))

import streamlit as st
from booking_agent_core import run_agent_turn, build_agent_for_session

st.set_page_config(
    page_title="Tidy Cuts — AI Booking Assistant Demo",
    page_icon="✂️",
    layout="centered",
)

with st.sidebar:
    st.markdown("### About this demo")
    st.markdown(
        "This is a working AI booking assistant built with LangChain, "
        "Claude, and RAG (retrieval-augmented generation).\n\n"
        "**It can:**\n"
        "- Check appointment availability and book a slot\n"
        "- Answer real policy questions (cancellation, deposits, services, "
        "hours) by retrieving actual business documents — not guessing\n"
        "- Escalate to a human for pricing, refunds, or anything it's not "
        "authorized to handle\n\n"
        "This demo uses a fictional salon (Tidy Cuts) with mock data. "
        "**Your bookings here are private to this browser session** — "
        "other visitors trying this demo won't see or affect your test "
        "data, and nothing is saved after you close this tab.\n\n"
        "A real build connects to your actual calendar, branding, and "
        "policies."
    )
    st.divider()
    st.markdown("### Try asking:")
    st.markdown(
        "- *\"What's available on Tuesday?\"*\n"
        "- *\"Do I need a deposit for balayage?\"*\n"
        "- *\"What's your cancellation policy?\"*\n"
        "- *\"How much does a haircut cost?\"* (watch it escalate, not guess)"
    )
    st.divider()
    if st.button("Reset my conversation"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.session_bookings = []
        st.rerun()


st.title("✂️ Tidy Cuts")
st.caption("AI booking assistant — demo build")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_bookings" not in st.session_state:
    st.session_state.session_bookings = []

if "tools_by_name" not in st.session_state or "bound_model" not in st.session_state:
    tools_by_name, bound_model = build_agent_for_session(st.session_state.session_bookings)
    st.session_state.tools_by_name = tools_by_name
    st.session_state.bound_model = bound_model

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about booking, hours, policies...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, terminal_event = run_agent_turn(
                st.session_state.chat_history,
                user_input,
                st.session_state.tools_by_name,
                st.session_state.bound_model,
            )
        st.markdown(reply)

        if terminal_event == "booked":
            st.success("✅ Appointment booked! (saved only for this session)")
        elif terminal_event == "escalated":
            st.info("🙋 This has been escalated to a human team member.")

    st.session_state.messages.append({"role": "assistant", "content": reply})