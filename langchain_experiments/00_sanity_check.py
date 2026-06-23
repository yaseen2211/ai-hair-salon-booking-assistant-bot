"""
Layer 2, Step 0: Sanity check.

Confirms langchain-anthropic is installed correctly and can reach the
Claude API using the same .env key as Layer 1. Run this first — if it
fails, fix it before moving on to the real LCEL experiments.

Run: PYTHONPATH=scripts python langchain_experiments/00_sanity_check.py
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=200,
)

if __name__ == "__main__":
    response = model.invoke("Say 'LangChain is wired up correctly' and nothing else.")
    print(f"Response type: {type(response)}")
    print(f"Response content: {response.content}")