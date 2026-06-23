"""
Experiment 2a: ZERO-SHOT structured extraction.

No examples given — just instructions and a schema. This is the baseline
to compare against few-shot and chain-of-thought versions.

Run: PYTHONPATH=scripts python prompts/extraction_zero_shot.py
"""

EXTRACTION_PROMPT_ZERO_SHOT = """You extract structured booking information from customer messages.

Always respond with ONLY a JSON object, no other text, in this exact shape:
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
- notes should capture anything relevant that doesn't fit the other fields
  (special requests, concerns, accessibility needs). Use null if nothing extra.
- If a field isn't mentioned, use null. Do not guess or invent information.
"""

TEST_INPUTS = [
    "hi! wondering if yall have anything open early next week, mornings work best, "
    "oh and also could someone with experience doing curly hair help me, it's my first time at a new salon",
    "Tuesday or Wednesday, whichever, just need a quick trim before a work thing",
    "I want to come in Saturday evening for color correction, I tried to bleach my own hair "
    "and it went badly, I've been to Tidy Cuts before for cuts but never color",
]

def strip_code_fences(raw_text: str) -> str:
    """Defensively strip ```json ... ``` wrappers if the model adds them
    despite being told not to. This is a real, common defensive-parsing
    pattern — instructions alone don't guarantee format compliance."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop the ```json line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


if __name__ == "__main__":
    import json
    from claude_client import ask_claude

    for user_input in TEST_INPUTS:
        print(f"\n{'-'*60}")
        print(f"CUSTOMER: {user_input}")
        reply = ask_claude(EXTRACTION_PROMPT_ZERO_SHOT, user_input)
        print(f"RAW OUTPUT: {reply}")
        cleaned = strip_code_fences(reply)
        try:
            parsed = json.loads(cleaned)
            print(f"PARSED OK (after stripping fences): {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"!! STILL FAILED TO PARSE: {e}")