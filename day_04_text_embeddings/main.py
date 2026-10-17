"""
Day 04: Text Embeddings (Concepts and Models)
Core Concept: Translating human meaning into mathematical coordinates
Chapter: Foundations

This script executes the key algorithmic implementations and examples from Day 4.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 04: Text Embeddings (Concepts and Models)")
    print("=" * 60)

# --- Snippet 1 ---
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials.",
    "What is the home loan interest rate?",
    "How much does it cost to borrow money to buy a house?",
    "The weather is sunny today.",
]

emb = model.encode(sentences, normalize_embeddings=True)
print("Shape:", emb.shape)                         # (5, 384)
print("Vector length:", np.linalg.norm(emb[0]).round(3))  # 1.0 (normalized)

sim = emb @ emb.T                                  # cosine similarity (vectors are normalized)
np.set_printoptions(precision=2, suppress=True)
print(sim)

# --- Snippet 2 ---
texts = [
    "I deposited my salary at the bank.",
    "The financial institution approved my loan.",
    "We had a picnic on the river bank.",
    "The children played near the edge of the stream.",
]
e = model.encode(texts, normalize_embeddings=True)
labels = ["bank(money)", "institution", "bank(river)", "stream edge"]
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        print(f"{labels[i]:>12} vs {labels[j]:<12}: {e[i] @ e[j]:.2f}")

# --- Snippet 3 ---
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedder:
    """Drop-in replacement for TfidfEmbedder (for models that need no prefixes, e.g., MiniLM)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        pass  # pre-trained, so there's nothing to fit (unlike TF-IDF)

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

# --- Snippet 4 ---
class PrefixAwareEmbedder:
    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = ""):
        self.model = SentenceTransformer(model_name)
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix

    def embed_documents(self, texts):
        return self.model.encode([self.doc_prefix + t for t in texts], normalize_embeddings=True)

    def embed_query(self, text):
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True)

# --- Snippet 5 ---
# Re-use DOCS, WordChunker, InMemoryVectorStore, EchoGenerator, RAGPipeline,
# EVAL_SET and evaluate_retrieval from Day 3.

for name, embedder, min_score in [
    ("TF-IDF", TfidfEmbedder(), 0.05),
    ("MiniLM", SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"), 0.2),
]:
    rag = RAGPipeline(WordChunker(25, 5), embedder, InMemoryVectorStore(), EchoGenerator(),
                      k=3, min_score=min_score)
    rag.index(DOCS)
    print(f"\n=== {name} ===")
    print(evaluate_retrieval(rag.retriever, EVAL_SET, k=3, filters={"year": 2026}))

# --- Snippet 6 ---
from sentence_transformers import SentenceTransformer

MODELS = [
    # (model name, query prefix, doc prefix)
    ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    ("intfloat/e5-small-v2", "query: ", "passage: "),
]

chunks = [c for d in DOCS if d.metadata["year"] == 2026 for c in WordChunker(25, 5).split(d)]
chunk_texts = [c.text for c in chunks]

def recall_mrr(model_name, qp, dp, k=3):
    m = SentenceTransformer(model_name)
    doc_vecs = m.encode([dp + t for t in chunk_texts], normalize_embeddings=True)
    hits, rr = 0, 0.0
    for question, relevant in EVAL_SET:
        q_vec = m.encode(qp + question, normalize_embeddings=True)
        top = np.argsort(-(doc_vecs @ q_vec))[:k]
        doc_ids = [chunks[i].doc_id for i in top]
        if relevant in doc_ids:
            hits += 1
            rr += 1 / (doc_ids.index(relevant) + 1)
    n = len(EVAL_SET)
    return round(hits / n, 2), round(rr / n, 2)

for name, qp, dp in MODELS:
    r, mrr = recall_mrr(name, qp, dp)
    print(f"{name:<45} Recall@3={r}  MRR={mrr}")

# --- Snippet 7 ---
pairs = [
    ("Your loan has been approved.", "Your loan has not been approved."),        # negation
    ("The home loan rate is 7.9%.", "The home loan rate is 8.6%."),             # numbers
    ("IFSC code ZYNT0001234", "IFSC code ZYNT0009876"),                         # IDs
    ("Transfer money to my savings account", "Transfer money from my savings account"), # direction
    ("Your loan has been approved.", "Congratulations, the credit was sanctioned."),   # paraphrase
]
for a, b in pairs:
    va, vb = model.encode([a, b], normalize_embeddings=True)
    print(f"{va @ vb:.2f} | {a} <-> {b}")

# --- Snippet 8 ---
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

texts = [
    # Loans
    "home loan interest rate", "mortgage EMI calculation", "cost of borrowing for a house",
    # Cards
    "credit card annual fee", "cashback on credit card", "lost my debit card",
    # Deposits
    "fixed deposit rates", "term deposit for senior citizens", "savings account interest",
    # Unrelated
    "football world cup", "best pizza in town", "weather forecast tomorrow",
]
groups = ["loans"] * 3 + ["cards"] * 3 + ["deposits"] * 3 + ["other"] * 3
colors = {"loans": "tab:blue", "cards": "tab:orange", "deposits": "tab:green", "other": "tab:gray"}

vecs = model.encode(texts, normalize_embeddings=True)
xy = PCA(n_components=2).fit_transform(vecs)

plt.figure(figsize=(8, 6))
for (x, y), t, g in zip(xy, texts, groups):
    plt.scatter(x, y, c=colors[g])
    plt.annotate(t, (x, y), fontsize=8)
plt.title("Embeddings projected to 2D (PCA)")
plt.show()

# --- Snippet 9 ---
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def openai_embed(texts, dims=None):
    kwargs = {"model": "text-embedding-3-small", "input": texts}
    if dims:
        kwargs["dimensions"] = dims
    resp = client.embeddings.create(**kwargs)
    vecs = np.array([d.embedding for d in resp.data])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

texts = ["What is the home loan interest rate?",
         "How much does it cost to borrow money to buy a house?",
         "The weather is sunny today."]

for dims in (None, 512, 256, 64):
    v = openai_embed(texts, dims)
    print(f"dims={v.shape[1]:>4}  loan-vs-house={v[0] @ v[1]:.3f}  loan-vs-weather={v[0] @ v[2]:.3f}")

# --- Snippet 10 ---
def embedding_storage_gb(num_chunks: int, dims: int, bytes_per_dim: float = 4) -> float:
    return num_chunks * dims * bytes_per_dim / 1024**3

def embedding_api_cost(num_chunks: int, avg_tokens_per_chunk: int,
                       price_per_million_tokens: float) -> float:
    return num_chunks * avg_tokens_per_chunk / 1_000_000 * price_per_million_tokens

for dims, bpd, label in [(1536, 4, "1536 float32"), (384, 4, "384 float32"), (384, 1, "384 int8")]:
    print(f"{label:<14}: {embedding_storage_gb(1_000_000, dims, bpd):.2f} GB for 1M chunks")

# Fill in the current price from your provider's pricing page
print("API cost:", embedding_api_cost(1_000_000, 400, price_per_million_tokens=0.02), "USD")

if __name__ == "__main__":
    print("Starting Day 04 script execution...")
