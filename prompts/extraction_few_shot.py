"""
Experiment 2b: FEW-SHOT structured extraction.

Same task as zero-shot, but with 3 worked examples included before the
real input. Compare output consistency and edge-case handling against
extraction_zero_shot.py using the SAME test inputs.

Run: PYTHONPATH=scripts python prompts/extraction_few_shot.py
"""

EXTRACTION_PROMPT_FEW_SHOT = """You extract structured booking information from customer messages.

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

Examples:

Customer message: "hey can i get a haircut next tuesday afternoon, first time here, i have a stubborn cowlick if that matters"
Output: {"service_type": "haircut", "preferred_date": "next Tuesday", "preferred_time_window": "afternoon", "is_first_time": true, "notes": "has a stubborn cowlick"}

Customer message: "looking to book something for this weekend, not sure what yet, maybe a trim"
Output: {"service_type": "trim", "preferred_date": "this weekend", "preferred_time_window": null, "is_first_time": null, "notes": "not sure exactly what service yet"}

Customer message: "balayage please, I've been a client for years"
Output: {"service_type": "balayage", "preferred_date": null, "preferred_time_window": null, "is_first_time": false, "notes": null}
"""

# Same inputs as extraction_zero_shot.py on purpose — this is what makes the comparison valid
TEST_INPUTS = [
    "hi! wondering if yall have anything open early next week, mornings work best, "
    "oh and also could someone with experience doing curly hair help me, it's my first time at a new salon",
    "Tuesday or Wednesday, whichever, just need a quick trim before a work thing",
    "I want to come in Saturday evening for color correction, I tried to bleach my own hair "
    "and it went badly, I've been to Tidy Cuts before for cuts but never color",
]

if __name__ == "__main__":
    import json
    from claude_client import ask_claude

    for user_input in TEST_INPUTS:
        print(f"\n{'-'*60}")
        print(f"CUSTOMER: {user_input}")
        reply = ask_claude(EXTRACTION_PROMPT_FEW_SHOT, user_input)
        print(f"RAW OUTPUT: {reply}")
        try:
            parsed = json.loads(reply)
            print(f"PARSED OK: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"!! FAILED TO PARSE AS JSON: {e}")