"""
Day 01: LLM Basics (Tokens, Context Window, Prompting)
Core Concept: The engine under the hood: tokens, context windows, and prompts
Chapter: Foundations

This script executes the key algorithmic implementations and examples from Day 1.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 01: LLM Basics (Tokens, Context Window, Prompting)")
    print("=" * 60)

# --- Snippet 1 ---
import tiktoken

enc = tiktoken.get_encoding("o200k_base") # used by GPT-4o family

text = "Retrieval-Augmented Generation improves LLM accuracy."
tokens = enc.encode(text)

print("Token count:", len(tokens))
print("Token IDs:", tokens)
print("Tokens:", [enc.decode([t]) for t in tokens])

# --- Snippet 2 ---
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tok.tokenize("Tokenization is powerful!"))
# ['token', '##ization', 'is', 'powerful', '!']

# --- Snippet 3 ---
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) # never hard-code keys

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=200,
    messages=[
        {"role": "system", "content": "You are a concise financial tutor."},
        {"role": "user", "content": "Explain what an ETF is in 3 sentences."},
    ],
)

print(response.choices[0].message.content)
print("Tokens used:", response.usage)

# --- Snippet 4 ---
import ollama

resp = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain what an ETF is in 3 sentences."}],
    options={"temperature": 0.2},
)
print(resp["message"]["content"])

# --- Snippet 5 ---
context = """
[Doc 1] Acme Bank's Q2 2026 net profit was $4.2 billion, up 12% YoY.
[Doc 2] Acme Bank's CEO is Jane Doe, appointed in 2024.
"""

question = "What was Acme Bank's Q2 2026 net profit and who is the CEO?"

prompt = f"""Answer ONLY from the context. If not found, say "I don't know".
Cite the doc IDs.

CONTEXT:
\"\"\"{context}\"\"\"

QUESTION: {question}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    messages=[{"role": "user", "content": prompt}],
)
print(response.choices[0].message.content)

if __name__ == "__main__":
    print("Starting Day 01 script execution...")
