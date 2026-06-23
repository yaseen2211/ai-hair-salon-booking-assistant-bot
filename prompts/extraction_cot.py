"""
Experiment 2c: CHAIN-OF-THOUGHT structured extraction.

The model reasons step-by-step in <thinking> tags BEFORE producing the
JSON, instead of jumping straight to the answer. This tends to help most
on genuinely ambiguous inputs (e.g. conflicting info, implied-but-not-stated
fields) rather than simple ones.

Note the output is no longer pure JSON — we have to extract the JSON block
from inside the longer response. This is a real, common parsing challenge
worth experiencing firsthand.

Run: PYTHONPATH=scripts python prompts/extraction_cot.py
"""

EXTRACTION_PROMPT_COT = """You extract structured booking information from customer messages.

First, think step by step inside <thinking> tags: identify each piece of
information the customer gave, note anything ambiguous or conflicting, and
decide how to resolve it.

Then output a JSON object inside <answer> tags, in this exact shape:
{
  "service_type": string or null,
  "preferred_date": string or null,
  "preferred_time_window": string or null,
  "is_first_time": true, false, or null,
  "notes": string or null
}

Rules:
- preferred_date should be a relative description as the customer said it
  (e.g. "next Tuesday", "this weekend") — do not calculate actual calendar dates.
- preferred_time_window should be one of: "morning", "afternoon", "evening", or null.
- notes should capture anything relevant that doesn't fit the other fields.
- If a field isn't mentioned, use null. Do not guess or invent information.
- If information conflicts (e.g. customer mentions two different days),
  note the conflict in <thinking> and put the LAST stated preference in
  the final answer, mentioning the conflict in "notes".

Respond with <thinking>...</thinking> followed by <answer>...</answer>. Nothing else.
"""

# Deliberately includes an ambiguous/conflicting case (the 3rd one) where
# chain-of-thought should show a visible advantage over zero-shot.
TEST_INPUTS = [
    "hi! wondering if yall have anything open early next week, mornings work best, "
    "oh and also could someone with experience doing curly hair help me, it's my first time at a new salon",
    "Tuesday or Wednesday, whichever, just need a quick trim before a work thing",
    "can I come Monday morning? actually wait, I just remembered I have a thing then, "
    "let's say Friday afternoon instead. it's for a haircut, and I've been going to "
    "this salon for ages but never with this particular stylist",
]


def extract_json_block(raw_text: str) -> str:
    """Pull the content between <answer> tags out of the full response."""
    start = raw_text.find("<answer>") + len("<answer>")
    end = raw_text.find("</answer>")
    return raw_text[start:end].strip()


if __name__ == "__main__":
    import json
    from claude_client import ask_claude

    for user_input in TEST_INPUTS:
        print(f"\n{'-'*60}")
        print(f"CUSTOMER: {user_input}")
        reply = ask_claude(EXTRACTION_PROMPT_COT, user_input, max_tokens=1500)
        print(f"\nFULL RESPONSE (reasoning + answer):\n{reply}")

        json_str = extract_json_block(reply)
        try:
            parsed = json.loads(json_str)
            print(f"\nEXTRACTED JSON: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"\n!! FAILED TO PARSE EXTRACTED BLOCK: {e}")
            print(f"Block was: {json_str!r}")