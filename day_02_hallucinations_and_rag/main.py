"""
Day 02: Why LLMs Hallucinate and How RAG Helps
Core Concept: Grounding fluent guesses with verifiable external truth
Chapter: Foundations

This script executes the key algorithmic implementations and examples from Day 2.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 02: Why LLMs Hallucinate and How RAG Helps")
    print("=" * 60)

# --- Snippet 1 ---
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o-mini"

def ask(prompt: str, system: str = "You are a helpful assistant.", temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

# --- Snippet 2 ---
questions = [
    "What was ZyntraCorp Bank's net profit in Q2 2026?",              # fictional
    "Summarize the paper 'Quantum Ledger Networks' by R. Patel (2021).", # likely fictional
    "Why did Apple acquire Netflix in 2021?",                         # false premise
    "List 3 court cases about AI chatbots in India with citations.",
]

for q in questions:
    print("Q:", q)
    print("A:", ask(q, temperature=0.9), "\n" + "-" * 80)

# --- Snippet 3 ---
SAFE_SYSTEM = """You are a careful assistant.
- If you are not certain a fact is true, say "I'm not sure."
- Never invent names, numbers, citations, or URLs.
- If a question contains a false assumption, point it out."""

for q in questions:
    print("Q:", q)
    print("A:", ask(q, system=SAFE_SYSTEM), "\n" + "-" * 80)

# --- Snippet 4 ---
import re
from collections import Counter

# 1) Knowledge base (pretend these are your company docs)
DOCS = [
    {"id": "DOC-1", "text": "ZyntraCorp Bank reported a net profit of $1.8 billion in Q2 2026, up 9% year over year."},
    {"id": "DOC-2", "text": "ZyntraCorp Bank's CEO is Priya Menon, appointed in March 2025."},
    {"id": "DOC-3", "text": "The ZyntraCorp Savings Plus account offers 4.25% annual interest with no minimum balance."},
    {"id": "DOC-4", "text": "ZyntraCorp Bank's home loan interest rate starts at 7.9% for salaried customers."},
    {"id": "DOC-5", "text": "Customer support is available 24/7 via phone at 1800-000-000 and in-app chat."},
]

STOPWORDS = {"the", "is", "a", "an", "of", "in", "what", "who", "for", "and", "to", "at", "with", "s"}

def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9.%]+", text.lower()) if w not in STOPWORDS]

# 2) Retriever: simple keyword-overlap score
def retrieve(query: str, k: int = 2) -> list[dict]:
    q_tokens = Counter(tokenize(query))
    scored = []
    for doc in DOCS:
        d_tokens = Counter(tokenize(doc["text"]))
        score = sum((q_tokens & d_tokens).values())
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:k] if score > 0]

# 3) Augment + Generate
RAG_SYSTEM = """Answer ONLY using the provided context.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
Cite the document ID(s) in square brackets, e.g., [DOC-1]."""

def rag_answer(query: str, k: int = 2) -> str:
    hits = retrieve(query, k)
    context = "\n".join([f"[{d['id']}] {d['text']}" for d in hits]) or "(no relevant documents)"
    prompt = f'CONTEXT:\n"""\n{context}\n"""\n\nQUESTION: {query}'
    return ask(prompt, system=RAG_SYSTEM)

# 4) Compare with and without RAG
tests = [
    "What was ZyntraCorp Bank's net profit in Q2 2026?",
    "Who is the CEO of ZyntraCorp Bank?",
    "What is the savings account interest rate at ZyntraCorp?",
    "What is ZyntraCorp Bank's stock price today?",  # not in the KB
]

for q in tests:
    print("Q:", q)
    print("  ❌ No RAG :", ask(q))
    print("  ✅ RAG    :", rag_answer(q))
    print("-" * 80)

# --- Snippet 5 ---
# The user's wording doesn't match the document wording
q = "Who runs ZyntraCorp?"          # the doc says "CEO", not "runs"
print("Retrieved:", [d["id"] for d in retrieve(q)])
print("Answer   :", rag_answer(q))

# --- Snippet 6 ---
def self_consistency(question: str, n: int = 5) -> list[str]:
    return [ask(question + " Answer in one short sentence.", temperature=1.0) for _ in range(n)]

for q in ["What is the capital of Australia?", "What was ZyntraCorp Bank's Q2 2026 net profit?"]:
    print("Q:", q)
    for a in self_consistency(q):
        print("  -", a)
    print()

# --- Snippet 7 ---
JUDGE_SYSTEM = """You are a strict fact-checker.
Given CONTEXT and ANSWER, reply with JSON only:
{"faithful": true|false, "unsupported_claims": ["..."]}"""

def check_faithfulness(context: str, answer: str) -> str:
    prompt = f'CONTEXT:\n"""{context}"""\n\nANSWER:\n"""{answer}"""'
    return ask(prompt, system=JUDGE_SYSTEM)

ctx = DOCS[0]["text"]
print(check_faithfulness(ctx, "ZyntraCorp's Q2 2026 profit was $1.8B, up 9%."))              # faithful
print(check_faithfulness(ctx, "ZyntraCorp's Q2 2026 profit was $1.8B, driven by crypto."))   # unsupported

if __name__ == "__main__":
    print("Starting Day 02 script execution...")
