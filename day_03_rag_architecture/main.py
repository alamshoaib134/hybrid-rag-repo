"""
Day 03: RAG Architecture (Retriever, Generator, Pipeline Flow)
Core Concept: The end-to-end blueprint: indexing, retrieval, and generation
Chapter: Foundations

This script executes the key algorithmic implementations and examples from Day 3.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 03: RAG Architecture (Retriever, Generator, Pipeline Flow)")
    print("=" * 60)

# --- Snippet 1 ---
# rag_from_scratch.py
import os
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------- Data models -----------------------
@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict


# ----------------------- 1. Chunker -----------------------
class WordChunker:
    """Fixed-size chunks measured in words, with overlap."""

    def __init__(self, chunk_size: int = 40, overlap: int = 10):
        assert 0 <= overlap < chunk_size
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        words = doc.text.split()
        step = self.chunk_size - self.overlap
        chunks = []
        for i, start in enumerate(range(0, len(words), step)):
            piece = words[start : start + self.chunk_size]
            chunks.append(
                Chunk(
                    id=f"{doc.id}#c{i}",
                    doc_id=doc.id,
                    text=" ".join(piece),
                    metadata={**doc.metadata, "chunk_index": i},
                )
            )
            if start + self.chunk_size >= len(words):
                break
        return chunks


# ----------------------- 2. Embedder -----------------------
class TfidfEmbedder:
    """Stand-in for a neural embedder. Vectors are L2-normalized, so dot product = cosine similarity."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()


# ----------------------- 3. Vector store -----------------------
class InMemoryVectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vec: np.ndarray, k: int = 3, filters: dict | None = None):
        scores = self.vectors @ query_vec
        results = []
        for idx in np.argsort(-scores):
            chunk = self.chunks[idx]
            if filters and any(chunk.metadata.get(key) != val for key, val in filters.items()):
                continue
            results.append((chunk, float(scores[idx])))
            if len(results) == k:
                break
        return results


# ----------------------- 4. Retriever -----------------------
class Retriever:
    def __init__(self, embedder: TfidfEmbedder, store: InMemoryVectorStore, min_score: float = 0.05):
        self.embedder = embedder
        self.store = store
        self.min_score = min_score

    def retrieve(self, query: str, k: int = 3, filters: dict | None = None):
        q_vec = self.embedder.embed([query])[0]
        hits = self.store.search(q_vec, k=k, filters=filters)
        return [(c, s) for c, s in hits if s >= self.min_score]


# ----------------------- 5. Prompt builder -----------------------
SYSTEM_PROMPT = """You are a banking assistant.
Answer ONLY using the CONTEXT. If the answer is not in the CONTEXT, reply exactly:
"I don't know based on the provided documents."
Cite chunk IDs in square brackets, e.g., [LOANS#c0]. Be concise."""


class PromptBuilder:
    def build(self, question: str, hits) -> list[dict]:
        if hits:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c, _ in hits)
        else:
            context = "(no relevant documents found)"
        user = f"CONTEXT:\n\"\"\"\n{context}\n\"\"\"\n\nQUESTION: {question}"
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ----------------------- 6. Generators -----------------------
class OpenAIGenerator:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, messages=messages
        )
        return resp.choices[0].message.content


class EchoGenerator:
    """Offline stand-in: shows the prompt the LLM would receive (no API key needed)."""

    def generate(self, messages: list[dict]) -> str:
        return f"[ECHO] Prompt sent to LLM:\n" + messages[-1]["content"]


# ----------------------- 7. Pipeline -----------------------
class RAGPipeline:
    def __init__(self, chunker, embedder, store, generator, k: int = 3, min_score: float = 0.05):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.retriever = Retriever(embedder, store, min_score)
        self.prompt_builder = PromptBuilder()
        self.generator = generator
        self.k = k

    # Indexing (offline)
    def index(self, docs: list[Document]) -> None:
        chunks = [c for d in docs for c in self.chunker.split(d)]
        self.embedder.fit([c.text for c in chunks])  # TF-IDF needs fitting; neural embedders don't
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        print(f"Indexed {len(docs)} docs -> {len(chunks)} chunks -> vectors {vectors.shape}")

    # Query (online)
    def query(self, question: str, filters: dict | None = None) -> dict:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, k=self.k, filters=filters)
        t1 = time.perf_counter()
        messages = self.prompt_builder.build(question, hits)
        answer = self.generator.generate(messages)
        t2 = time.perf_counter()
        return {
            "question": question,
            "answer": answer,
            "sources": [(c.id, round(s, 3)) for c, s in hits],
            "timing_ms": {"retrieve": round((t1 - t0) * 1000, 1), "generate": round((t2 - t1) * 1000, 1)},
        }

# --- Snippet 2 ---
DOCS = [
    Document("LOANS", "ZyntraCorp Bank home loan interest rates start at 7.9% per annum for salaried "
                      "customers and 8.3% for self-employed customers. The maximum tenure is 30 years. "
                      "Processing fees are 0.5% of the loan amount, capped at 10,000 rupees.",
             {"department": "retail_lending", "year": 2026}),
    Document("SAVINGS", "The Savings Plus account offers 4.25% annual interest, credited quarterly. "
                        "There is no minimum balance requirement. Customers get free unlimited UPI transfers "
                        "and a free debit card for the first year.",
             {"department": "retail_banking", "year": 2026}),
    Document("CARDS", "The Platinum credit card has an annual fee of 2,500 rupees, waived on annual "
                      "spends above 3 lakh rupees. It offers 2% cashback on online purchases and airport "
                      "lounge access four times per year.",
             {"department": "cards", "year": 2026}),
    Document("FD", "Fixed deposit rates are 7.1% for tenures of 1 to 2 years and 7.4% for senior "
                   "citizens. Premature withdrawal attracts a penalty of 1% on the applicable rate.",
             {"department": "retail_banking", "year": 2026}),
    Document("SUPPORT", "Customer support is available 24/7 via phone at 1800-000-000 and through "
                        "in-app chat. Lost card reports are handled immediately through the mobile app.",
             {"department": "operations", "year": 2026}),
    Document("LOANS-OLD", "ZyntraCorp Bank home loan interest rates start at 8.6% per annum for all "
                          "customers. This policy applied during 2023.",
             {"department": "retail_lending", "year": 2023}),
]

# Swap EchoGenerator() -> OpenAIGenerator() when you have an API key
rag = RAGPipeline(
    chunker=WordChunker(chunk_size=40, overlap=10),
    embedder=TfidfEmbedder(),
    store=InMemoryVectorStore(),
    generator=EchoGenerator(),
    k=3,
)

rag.index(DOCS)

for q in [
    "What is the home loan interest rate for salaried customers?",
    "Is there a minimum balance for the savings account?",
    "What is the FD rate for senior citizens?",
    "What is ZyntraCorp's share price?",  # not in the KB
]:
    result = rag.query(q)
    print(f"\nQ: {result['question']}")
    print(f"Sources: {result['sources']}")
    print(f"Timing : {result['timing_ms']}")
    print(f"Answer : {result['answer'][:500]}")

# --- Snippet 3 ---
q = "What is the home loan interest rate?"
print("No filter :", rag.query(q)["sources"])
print("year=2026 :", rag.query(q, filters={"year": 2026})["sources"])

# --- Snippet 4 ---
EVAL_SET = [
    ("home loan rate for salaried people", "LOANS"),
    ("processing fee on housing loan", "LOANS"),
    ("savings account interest rate", "SAVINGS"),
    ("do I need to keep minimum balance", "SAVINGS"),
    ("credit card annual fee waiver", "CARDS"),
    ("lounge access on card", "CARDS"),
    ("fixed deposit rate senior citizens", "FD"),
    ("penalty for breaking FD early", "FD"),
    ("how do I report a lost card", "SUPPORT"),
    ("customer care phone number", "SUPPORT"),
]

def evaluate_retrieval(retriever: Retriever, eval_set, k: int = 3, filters: dict | None = None) -> dict:
    hits_count, reciprocal_ranks = 0, []
    for question, relevant_doc in eval_set:
        doc_ids = [c.doc_id for c, _ in retriever.retrieve(question, k=k, filters=filters)]
        if relevant_doc in doc_ids:
            hits_count += 1
            reciprocal_ranks.append(1 / (doc_ids.index(relevant_doc) + 1))
        else:
            reciprocal_ranks.append(0.0)
            print(f"  ❌ MISS: '{question}' -> got {doc_ids}")
    n = len(eval_set)
    return {"recall@k": round(hits_count / n, 2), "mrr": round(sum(reciprocal_ranks) / n, 2)}

for k in (1, 3, 5):
    print(f"k={k}:", evaluate_retrieval(rag.retriever, EVAL_SET, k=k, filters={"year": 2026}))

# --- Snippet 5 ---
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def fit_to_budget(hits, max_context_tokens: int):
    selected, used = [], 0
    for chunk, score in hits:  # hits are already sorted by relevance
        n = count_tokens(chunk.text)
        if used + n > max_context_tokens:
            break
        selected.append((chunk, score))
        used += n
    return selected, used

hits = rag.retriever.retrieve("home loan rate and processing fee", k=5)
for budget in (50, 100, 300):
    sel, used = fit_to_budget(hits, budget)
    print(f"budget={budget:>3} -> kept {len(sel)} chunks, {used} tokens: {[c.id for c, _ in sel]}")

if __name__ == "__main__":
    print("Starting Day 03 script execution...")
