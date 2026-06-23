"""
Layer 2, Experiment 1b: FEW-SHOT extraction, rebuilt with LCEL.

Same task as 01a, but the system template now includes 3 worked examples,
same as Layer 1's prompts/extraction_few_shot.py. Compare format
consistency against the zero-shot version using the same test inputs.

New thing to notice here: the few-shot examples themselves contain literal
{ } JSON braces, just like the schema does — so they need the same {{ }}
escaping treatment throughout, not just in the schema block.

Run: python langchain_experiments/01b_extraction_few_shot.py
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def get_model(max_tokens: int = 1000, temperature: float = 0.0):
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

Examples:

Customer message: "hey can i get a haircut next tuesday afternoon, first time here, i have a stubborn cowlick if that matters"
Output: {{"service_type": "haircut", "preferred_date": "next Tuesday", "preferred_time_window": "afternoon", "is_first_time": true, "notes": "has a stubborn cowlick"}}

Customer message: "looking to book something for this weekend, not sure what yet, maybe a trim"
Output: {{"service_type": "trim", "preferred_date": "this weekend", "preferred_time_window": null, "is_first_time": null, "notes": "not sure exactly what service yet"}}

Customer message: "balayage please, I've been a client for years"
Output: {{"service_type": "balayage", "preferred_date": null, "preferred_time_window": null, "is_first_time": false, "notes": null}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("user", "{customer_message}"),
])

model = get_model(max_tokens=500)
chain = prompt | model | StrOutputParser()

TEST_INPUTS = [
    "hi! wondering if yall have anything open early next week, mornings work best, "
    "oh and also could someone with experience doing curly hair help me, it's my first time at a new salon",
    "Tuesday or Wednesday, whichever, just need a quick trim before a work thing",
    "I want to come in Saturday evening for color correction, I tried to bleach my own hair "
    "and it went badly, I've been to Tidy Cuts before for cuts but never color",
]


def strip_code_fences(raw_text: str) -> str:
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
        raw_output = chain.invoke({
            "business_name": "Tidy Cuts",
            "customer_message": user_input,
        })
        print(f"RAW OUTPUT: {raw_output}")
        cleaned = strip_code_fences(raw_output)
        try:
            parsed = json.loads(cleaned)
            print(f"PARSED OK: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"!! FAILED TO PARSE: {e}")