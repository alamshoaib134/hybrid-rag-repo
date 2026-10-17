"""
Day 05: Similarity Metrics (Cosine, Dot Product, Euclidean)
Core Concept: The geometry of relevance: Cosine, Dot Product, and Euclidean
Chapter: Foundations

This script executes the key algorithmic implementations and examples from Day 5.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 05: Similarity Metrics (Cosine, Dot Product, Euclidean)")
    print("=" * 60)

# --- Snippet 1 ---
import faiss, numpy as np

vecs = np.random.rand(1000, 384).astype("float32")
faiss.normalize_L2(vecs)                # in-place normalization
index = faiss.IndexFlatIP(384)          # inner product = cosine on unit vectors
index.add(vecs)

q = np.random.rand(1, 384).astype("float32")
faiss.normalize_L2(q)
scores, ids = index.search(q, 5)        # higher score = more similar

# --- Snippet 2 ---
import chromadb

client = chromadb.Client()
col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
# Note: col.query(...) returns *distances* (cosine distance = 1 - cosine similarity)

# --- Snippet 3 ---
import numpy as np

def dot(a, b):
    return float(sum(x * y for x, y in zip(a, b)))

def norm(a):
    return dot(a, a) ** 0.5

def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def euclidean(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

A = np.array([3.0, 4.0])
B = np.array([4.0, 3.0])

# Using custom functions vs numpy
print("Custom:", round(dot(A, B), 4), round(cosine(A, B), 4), round(euclidean(A, B), 4))
print("NumPy: ", round(float(A @ B), 4),
      round(float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B))), 4),
      round(float(np.linalg.norm(A - B)), 4))

# --- Snippet 4 ---
q = np.array([1.0, 1.0])
docs = {
    "d_long ": np.array([5.0, 5.0]),
    "d_short": np.array([1.0, 1.0]),
    "d_tilt ": np.array([1.0, 0.8]),
}

def cos_np(x, y):
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

print(f"{'doc':<8} {'dot':>6} {'cosine':>7} {'L2':>6}")
for name, d in docs.items():
    print(f"{name:<8} {q @ d:>6.2f} {cos_np(q, d):>7.3f} {np.linalg.norm(q - d):>6.2f}")

for metric, fn, reverse in [
    ("dot", lambda d: q @ d, True),
    ("cosine", lambda d: cos_np(q, d), True),
    ("L2", lambda d: np.linalg.norm(q - d), False),
]:
    ranking = sorted(docs, key=lambda n: fn(docs[n]), reverse=reverse)
    print(f"{metric:>6} ranking: {[n.strip() for n in ranking]}")

# Now normalize everything and re-rank
unit = lambda v: v / np.linalg.norm(v)
qn = unit(q)
print("\nAfter normalization:")
for name, d in docs.items():
    dn = unit(d)
    print(f"{name:<8} dot={qn @ dn:.3f}  L2={np.linalg.norm(qn - dn):.3f}")

# --- Snippet 5 ---
rng = np.random.default_rng(42)

def normalize_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

a, b = normalize_rows(rng.normal(size=(2, 384)))
print("L2^2      :", round(float(np.sum((a - b) ** 2)), 6))
print("2 - 2*cos :", round(float(2 - 2 * (a @ b)), 6))

# Rankings are identical for normalized vectors
docs = normalize_rows(rng.normal(size=(1000, 384)))
q = normalize_rows(rng.normal(size=384))

top_dot = np.argsort(-(docs @ q))[:10]
top_l2 = np.argsort(np.linalg.norm(docs - q, axis=1))[:10]
print("Same top-10 (dot vs L2)?", np.array_equal(top_dot, top_l2))

# --- Snippet 6 ---
raw = rng.normal(size=(1000, 384)) * rng.uniform(0.5, 3.0, size=(1000, 1))  # random lengths
qr = rng.normal(size=384)

dot_scores = raw @ qr
cos_scores = dot_scores / (np.linalg.norm(raw, axis=1) * np.linalg.norm(qr))
l2_dists = np.linalg.norm(raw - qr, axis=1)

top = lambda arr, k=10, largest=True: set(np.argsort(-arr if largest else arr)[:k])
t_dot, t_cos, t_l2 = top(dot_scores), top(cos_scores), top(l2_dists, largest=False)

print("Top-10 overlap dot vs cosine:", len(t_dot & t_cos), "/ 10")
print("Top-10 overlap L2 vs cosine:", len(t_l2 & t_cos), "/ 10")
print("Avg length of dot's top-10  :", round(float(np.linalg.norm(raw[list(t_dot)], axis=1).mean()), 1))
print("Avg length of L2's top-10   :", round(float(np.linalg.norm(raw[list(t_l2)], axis=1).mean()), 1))
print("Avg length overall          :", round(float(np.linalg.norm(raw, axis=1).mean()), 1))

# --- Snippet 7 ---
import time

N, D = 100_000, 384
corpus = normalize_rows(rng.normal(size=(N, D)).astype(np.float32))
target = 12345
query = normalize_rows(corpus[target] + 0.02 * rng.normal(size=D).astype(np.float32))  # a "paraphrase"

# (a) Python loop over 10k rows
t0 = time.perf_counter()
loop_scores = [float(np.dot(v, query)) for v in corpus[:10_000]]
t_loop = time.perf_counter() - t0

# (b) One matrix-vector product over all 100k rows
t0 = time.perf_counter()
scores = corpus @ query
k = 5
top_k = np.argpartition(-scores, k)[:k]            # O(N): unordered top-k
top_k = top_k[np.argsort(-scores[top_k])]           # sort just those k
t_vec = time.perf_counter() - t0

print(f"Loop over 10k rows : {t_loop * 1000:.1f} ms")
print(f"Vectorized 100k    : {t_vec * 1000:.1f} ms")
print("Top-5 ids:", top_k.tolist(), "| scores:", [round(float(s), 3) for s in scores[top_k]])
print("Target found at rank 1?", top_k[0] == target)

# --- Snippet 8 ---
print(f"{'dims':>5} {'mean cos':>9} {'std':>7} {'min':>7} {'max':>7}  1/sqrt(d)")
for dim in (2, 10, 100, 384, 1536):
    x = normalize_rows(rng.normal(size=(4000, dim)))
    sims = np.sum(x[:2000] * x[2000:], axis=1)
    print(f"{dim:>5} {sims.mean():>9.3f} {sims.std():>7.3f} {sims.min():>7.3f} {sims.max():>7.3f}  {1/np.sqrt(dim):.3f}")

# --- Snippet 9 ---
def mmr(query_vec, doc_vecs, k=3, lambda_=0.5):
    sim_q = doc_vecs @ query_vec             # relevance to the query
    sim_d = doc_vecs @ doc_vecs.T           # similarity between docs
    selected = [int(np.argmax(sim_q))]
    candidates = set(range(len(doc_vecs))) - set(selected)
    while len(selected) < k and candidates:
        best = max(
            candidates,
            key=lambda i: lambda_ * sim_q[i] - (1 - lambda_) * max(sim_d[i, j] for j in selected),
        )
        selected.append(best)
        candidates.remove(best)
    return selected

labels = [
    "Home loan rate is 7.9% (policy v1)",
    "Home loan rate is 7.9% (policy v1 copy)",
    "Home loan rate is 7.9% (FAQ page)",
    "Home loan processing fee is 0.5%",
    "Home loan max tenure is 30 years",
    "Credit card annual fee is 2,500",
]

toy = normalize_rows(np.array([
    # dims: [loan-ness, rate, fee, tenure, card]
    [1.00, 0.50, 0.00, 0.00, 0.00],
    [1.00, 0.52, 0.02, 0.00, 0.00],
    [0.98, 0.50, 0.00, 0.03, 0.00],
    [0.90, 0.30, 0.50, 0.00, 0.00],
    [0.90, 0.30, 0.00, 0.50, 0.00],
    [0.05, 0.00, 0.00, 0.00, 1.00],
]))

q_toy = normalize_rows(np.array([1.0, 0.45, 0.25, 0.25, 0.0]))  # "Tell me about home loans"

print("Plain top-3 :", [labels[i] for i in np.argsort(-(toy @ q_toy))[:3]])
for lam in (1.0, 0.7, 0.5, 0.3):
    print(f"MMR lambda={lam:<4}:", [labels[i] for i in mmr(q_toy, toy, k=3, lambda_=lam)])

# --- Snippet 10 ---
def jaccard(a: str, b: str) -> float:
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / len(A | B)

pairs = [
    ("home loan interest rate", "home loan interest rates"),
    ("home loan interest rate", "cost of borrowing to buy a house"),
    ("loan approved", "loan not approved"),
]

for x, y in pairs:
    print(f"{jaccard(x, y):.2f} | {x!r} vs {y!r}")

# --- Snippet 11 ---
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
query = "cost of borrowing to buy a house"
texts = [
    "Home loan interest rates start at 7.9% per annum.",
    "Processing fees are 0.5% of the loan amount.",
    "The Platinum credit card has an annual fee of 2,500 rupees.",
    "Customer support is available 24/7.",
]

raw_docs, raw_q = model.encode(texts), model.encode(query)                 # un-normalized
print("Vector lengths:", np.round(np.linalg.norm(raw_docs, axis=1), 3))    # ~1.0 for this model

unit_docs, unit_q = normalize_rows(raw_docs), normalize_rows(raw_q)
for t, d_raw, d_unit in zip(texts, raw_docs, unit_docs):
    print(f"dot={raw_q @ d_raw:.3f}  cos={d_unit @ unit_q:.3f}  "
          f"L2={np.linalg.norm(unit_q - d_unit):.3f} | {t[:50]}")

if __name__ == "__main__":
    print("Starting Day 05 script execution...")
