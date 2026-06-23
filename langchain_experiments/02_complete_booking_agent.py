"""
Terminal chat version of the Tidy Cuts Booking Assistant.

Core agent logic lives in booking_agent_core.py — shared with the
Streamlit web version in app.py. This file builds one session's worth
of tools (since booking state is now per-session, not global) and adds
the interactive terminal loop on top.

Run from inside langchain_experiments/:
    cd langchain_experiments
    python 02_complete_booking_agent.py
"""

from booking_agent_core import run_agent_turn, build_agent_for_session

EXIT_PHRASES = {"exit", "quit", "bye", "never mind", "nevermind", "cancel", "stop"}


def chat():
    print("=" * 60)
    print("Tidy Cuts Booking Assistant — type 'exit' to leave anytime")
    print("=" * 60)
    print("(This terminal session has its own private bookings, separate")
    print(" from any other session — nothing is saved after you exit.)")

    # One session's worth of state, scoped to this single terminal run
    session_bookings = []
    tools_by_name, bound_model = build_agent_for_session(session_bookings)
    chat_history = []

    while True:
        user_input = input("\nYOU: ").strip()

        if not user_input:
            continue

        if user_input.lower() in EXIT_PHRASES:
            print("\nASSISTANT: No problem, reach out anytime if you'd like to book later!")
            print("\n[CONVERSATION ENDED: customer exited voluntarily]")
            break

        reply, terminal_event = run_agent_turn(
            chat_history, user_input, tools_by_name, bound_model
        )
        print(f"\nASSISTANT: {reply}")

        if terminal_event == "booked":
            print("\n[CONVERSATION ENDED: appointment booked]")
            print(f"Final booking record: {session_bookings[-1]}")
            break
        elif terminal_event == "escalated":
            print("\n[CONVERSATION ENDED: escalated to human]")
            break


if __name__ == "__main__":
    chat()