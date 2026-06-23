"""
Layer 2, Experiment 1a: ZERO-SHOT extraction, rebuilt with LCEL.

Same prompt content and same test inputs as Layer 1's
prompts/extraction_zero_shot.py — only the plumbing changes. Compare the
two directly: same outputs expected, different code shape.

Key LangChain concepts introduced here:
- ChatPromptTemplate: a reusable template with placeholders, instead of
  building one big string by hand each time.
- LCEL piping (`|`): chaining prompt -> model -> parser into one runnable.
- StrOutputParser: extracts the plain string content from the model's
  response object, so .invoke() returns a string instead of an AIMessage.
- Template variables: {business_name} is filled in at invoke time, so
  the same chain works for any business without editing the prompt text.

Run: python langchain_experiments/01a_extraction_zero_shot.py
(no PYTHONPATH needed — this file is self-contained)
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def get_model(max_tokens: int = 1000, temperature: float = 0.0):
    """Returns a LangChain chat model instance. Defined inline here so this
    script has no dependency on a separate model_config.py file."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=max_tokens,
        temperature=temperature,
    )


SYSTEM_TEMPLATE = """You extract structured booking information from customer messages
for {business_name}.

Always respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "service_type": string or null,
  "preferred_date": string or null,
  "preferred_time_window": string or null,
  "is_first_time": true, false, or null,
  "notes": string or null
}}

Rules:
- preferred_date should be a relative description as the customer said it
  (e.g. "next Tuesday", "this weekend") — do not calculate actual calendar dates.
- preferred_time_window should be one of: "morning", "afternoon", "evening", or null.
- notes should capture anything relevant that doesn't fit the other fields
  (special requests, concerns, accessibility needs). Use null if nothing extra.
- If a field isn't mentioned, use null. Do not guess or invent information.
"""

# {business_name} above is a REAL placeholder (single braces) — LangChain
# will substitute it from the dict passed to .invoke(). Contrast with the
# {{ }} double braces in the JSON schema, which are escaped literal braces,
# not placeholders. This distinction (single = fill in, double = literal)
# is the one tricky part of ChatPromptTemplate syntax worth internalizing.

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("user", "{customer_message}"),
])

model = get_model(max_tokens=500)

# This is the actual LCEL composition: prompt -> model -> parser
chain = prompt | model | StrOutputParser()

TEST_INPUTS = [
    "hi! wondering if yall have anything open early next week, mornings work best, "
    "oh and also could someone with experience doing curly hair help me, it's my first time at a new salon",
    "Tuesday or Wednesday, whichever, just need a quick trim before a work thing",
    "I want to come in Saturday evening for color correction, I tried to bleach my own hair "
    "and it went badly, I've been to Tidy Cuts before for cuts but never color",
]


def strip_code_fences(raw_text: str) -> str:
    """Same defensive parsing helper as Layer 1 — the model may still wrap
    output in ```json fences despite instructions, regardless of framework."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


if __name__ == "__main__":
    import json

    for user_input in TEST_INPUTS:
        print(f"\n{'-'*60}")
        print(f"CUSTOMER: {user_input}")
        # .invoke() runs the whole chain: fills the template, calls the
        # model, parses the output to a plain string, all in one call.
        raw_output = chain.invoke({
            "business_name": "Kabana Hairs",
            "customer_message": user_input,
        })
        print(f"RAW OUTPUT: {raw_output}")
        cleaned = strip_code_fences(raw_output)
        try:
            parsed = json.loads(cleaned)
            print(f"PARSED OK: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"!! FAILED TO PARSE: {e}")