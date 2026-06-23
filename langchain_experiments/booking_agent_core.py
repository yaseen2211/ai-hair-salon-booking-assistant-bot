"""
Core agent logic for the Tidy Cuts Booking Assistant.

Booking data is PER-SESSION, not global — each call to
build_agent_for_session() creates tools scoped to one session's own
bookings list, so simultaneous visitors never see or affect each
other's test bookings. Nothing is written to disk.
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import (
    policy_lookup,
    lookup_cancellation_policy,
    lookup_deposit_policy,
    lookup_services,
    lookup_hours_and_location,
)


# ============================================================
# MODEL SETUP
# ============================================================

def get_model(max_tokens: int = 1000, temperature: float = 0.0):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ============================================================
# FEW-SHOT TEMPLATE
# ============================================================

EXTRACTION_EXAMPLES = [
    {
        "input": "hey can i get a haircut next tuesday afternoon, first time here, i have a stubborn cowlick if that matters",
        "output": '{"service_type": "haircut", "preferred_date": "next Tuesday", "preferred_time_window": "afternoon", "is_first_time": true, "notes": "has a stubborn cowlick"}',
    },
    {
        "input": "looking to book something for this weekend, not sure what yet, maybe a trim",
        "output": '{"service_type": "trim", "preferred_date": "this weekend", "preferred_time_window": null, "is_first_time": null, "notes": "not sure exactly what service yet"}',
    },
    {
        "input": "balayage please, I've been a client for years",
        "output": '{"service_type": "balayage", "preferred_date": null, "preferred_time_window": null, "is_first_time": false, "notes": null}',
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=EXTRACTION_EXAMPLES,
)


# ============================================================
# PER-SESSION BOOKING TOOLS — no shared global, no file persistence
# ============================================================

MOCK_AVAILABILITY = {
    "Tuesday": ["10:00am", "2:00pm", "4:30pm"],
    "Wednesday": ["9:00am", "1:00pm"],
    "Friday": ["11:00am", "3:00pm"],
}


def make_check_availability_tool(session_bookings: list):
    """Returns a check_availability tool scoped to ONE session's bookings."""

    @tool
    def check_availability(day: str) -> str:
        """Check available appointment slots for Tidy Cuts on a given day
        of the week (e.g. 'Tuesday', 'Friday'). Returns the list of open
        times, or says none are available."""
        all_slots = MOCK_AVAILABILITY.get(day)
        if not all_slots:
            return (
                f"No available slots found for {day}. We only take bookings "
                f"Tuesday, Wednesday, or Friday — ask the customer to pick "
                f"one of those days instead."
            )
        taken = {b["time"] for b in session_bookings if b["day"] == day}
        open_slots = [s for s in all_slots if s not in taken]
        if not open_slots:
            return f"No remaining slots available on {day} in this demo session."
        return f"Available on {day}: {', '.join(open_slots)}"

    return check_availability


def make_book_appointment_tool(session_bookings: list):
    """Returns a book_appointment tool scoped to ONE session's bookings."""

    @tool
    def book_appointment(day: str, time: str, service_type: str, customer_note: str = "") -> str:
        """Book a confirmed appointment for Tidy Cuts. Only call this AFTER
        confirming the slot is available via check_availability, and only
        after the customer has explicitly agreed to the specific day and
        time."""
        booking = {
            "day": day,
            "time": time,
            "service_type": service_type,
            "note": customer_note,
            "booked_at": datetime.now().isoformat(),
        }
        session_bookings.append(booking)
        return f"Booked: {service_type} on {day} at {time}."

    return book_appointment


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human staff member. Use this when
    the customer asks for a refund, a price quote, becomes frustrated,
    or asks for something outside what you're authorized to handle."""
    print(f"  [SYSTEM: escalation logged — reason: {reason}]")
    return "Escalated to a human team member. They will follow up directly."


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a booking assistant for {business_name}, a small hair salon.

You have access to tools for checking availability, booking, escalating,
and looking up real policy/service information. You must use these tools
to take real action or get real information — you may NEVER simply claim
something is booked, checked, or escalated in text without actually
calling the corresponding tool first, and you may NEVER answer a policy
or service-description question from your own general knowledge.

Rules:
- Before booking, always call check_availability for the requested day.
- We only take bookings Tuesday, Wednesday, or Friday. If a customer asks
  for any other day, tell them clearly which days we DO take.
- Only call book_appointment after the customer has confirmed a specific
  available day and time.
- For policy or service questions, prefer the specific lookup tool that
  matches the topic (lookup_cancellation_policy, lookup_deposit_policy,
  lookup_services, lookup_hours_and_location). Only use the general
  policy_lookup if the question doesn't clearly fit one topic.
- Never quote a price, never promise a refund — call escalate_to_human
  for those topics instead.
- If a customer is insistent or frustrated after you've already declined
  something twice, call escalate_to_human rather than repeating yourself.
- Keep responses brief and friendly. No emojis.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    few_shot_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


# ============================================================
# PER-SESSION AGENT BUILDER
# ============================================================

def build_agent_for_session(session_bookings: list):
    """Builds a fresh set of tools and a bound model for ONE session's
    booking data. Call this once per session (e.g. once per Streamlit
    visitor), and reuse the returned objects for that session's turns."""

    check_availability = make_check_availability_tool(session_bookings)
    book_appointment = make_book_appointment_tool(session_bookings)

    tools = [
        check_availability,
        book_appointment,
        escalate_to_human,
        policy_lookup,
        lookup_cancellation_policy,
        lookup_deposit_policy,
        lookup_services,
        lookup_hours_and_location,
    ]
    tools_by_name = {t.name: t for t in tools}
    bound_model = get_model(max_tokens=1000).bind_tools(tools)

    return tools_by_name, bound_model


# ============================================================
# AGENT TURN
# ============================================================

def run_agent_turn(
    chat_history: list,
    user_input: str,
    tools_by_name: dict,
    bound_model,
    business_name: str = "Tidy Cuts",
):
    """Runs one full agent turn for ONE session. Returns (reply_text,
    terminal_event) where terminal_event is one of: None, "booked",
    "escalated". tools_by_name and bound_model must come from
    build_agent_for_session() for this session's booking data."""

    messages = prompt.invoke({
        "business_name": business_name,
        "chat_history": chat_history,
        "input": user_input,
    }).to_messages()

    chat_history.append(HumanMessage(content=user_input))
    terminal_event = None

    while True:
        response = bound_model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            chat_history.append(AIMessage(content=response.content))
            return response.content, terminal_event

        for tool_call in response.tool_calls:
            tool_fn = tools_by_name[tool_call["name"]]
            result = tool_fn.invoke(tool_call["args"])
            print(f"  [TOOL CALLED: {tool_call['name']}({tool_call['args']}) -> {result}]")
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            if tool_call["name"] == "book_appointment":
                terminal_event = "booked"
            elif tool_call["name"] == "escalate_to_human":
                terminal_event = "escalated"