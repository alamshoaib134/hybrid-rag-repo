"""
Day 13: Hybrid Search (Dense + Sparse)
Core Concept: Combining vector search and keyword search with Reciprocal Rank Fusion
Chapter: Search & Context

This script executes the key algorithmic implementations and examples from Day 13.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 13: Hybrid Search (Dense + Sparse)")
    print("=" * 60)

# --- Snippet 1 ---
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype="float32")

bm25 = BM25([analyze(t) for t in FULL_TEXT])       # from Day 12
doc_vecs = embed(FULL_TEXT)

# Mixed queries: code/name + natural paraphrase in the same question
ALL_QUERIES = QUERIES + [
    ("What is the SWIFT code for money sent to me from overseas?", "SWIFT", "mixed"),
    ("IFSC for the branch in Shivajinagar", "BR-PUNE", "mixed"),
    ("Can I pull surplus money out of ZYN-HL-FLEX?", "HL-FLEX", "mixed"),
    ("Tax relief on housing loan principal under 80C", "HL-TAX", "mixed"),
    ("What does E2091 mean for my daily payment limit?", "UPI-ERR", "mixed"),
]

def dense_scores(q: str, n: int = 10) -> dict[str, float]:
    s = doc_vecs @ embed([q])[0]
    return {IDS[i]: float(s[i]) for i in np.argsort(-s)[:n]}        # ordered best -> worst

def sparse_scores(q: str, n: int = 10) -> dict[str, float]:
    return {IDS[i]: s for i, s in bm25.search(analyze(q), k=n)}     # ordered best -> worst

def evaluate(search_fn, queries=ALL_QUERIES) -> dict[str, dict]:
    """search_fn(question) -> ranked list of doc IDs. Returns R@1, R@3, and MRR per query type."""
    by_type = defaultdict(list)
    for q, rel, qtype in queries:
        ranked = search_fn(q)
        rank = ranked.index(rel) + 1 if rel in ranked else None
        by_type[qtype].append(rank)
        by_type["ALL"].append(rank)
    return {t: {"R@1": np.mean([r == 1 for r in ranks]),
                "R@3": np.mean([r is not None and r <= 3 for r in ranks]),
                "MRR": np.mean([1 / r if r else 0 for r in ranks])}
            for t, ranks in by_type.items()}

def print_eval(name: str, res: dict) -> None:
    cells = "  ".join(f"{t}: R@1={m['R@1']:.2f} R@3={m['R@3']:.2f} MRR={m['MRR']:.2f}"
                      for t, m in res.items() if t in ("exact", "paraphrase", "mixed", "ALL"))
    print(f"{name:<22} {cells}")

print(len(ALL_QUERIES), "queries:", {t: sum(1 for *_, x in ALL_QUERIES if x == t) for t in ("exact", "paraphrase", "mixed")})

# --- Snippet 2 ---
q = "Can I pull surplus money out of ZYN-HL-FLEX?"
d, s = dense_scores(q, 5), sparse_scores(q, 5)
print("Dense  (cosine):", {k: round(v, 3) for k, v in d.items()})
print("Sparse (BM25)  :", {k: round(v, 2) for k, v in s.items()})

naive = defaultdict(float)
for doc, v in list(d.items()) + list(s.items()):
    naive[doc] += v
print("Naive sum      :", [(k, round(v, 2)) for k, v in sorted(naive.items(), key=lambda x: -x[1])[:5]])

# --- Snippet 3 ---
def rrf(ranked_lists: list[list[str]], k: int = 60, weights: list[float] | None = None) -> dict[str, float]:
    weights = weights or [1.0] * len(ranked_lists)
    scores = defaultdict(float)
    for w, lst in zip(weights, ranked_lists):
        for rank, doc in enumerate(lst, start=1):
            scores[doc] += w / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))

def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {d: 1.0 for d in scores}      # guard: one result or ties
    return {d: (v - lo) / (hi - lo) for d, v in scores.items()}

def zscore(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()))
    std = vals.std()
    return {d: (v - vals.mean()) / std if std > 0 else 0.0 for d, v in scores.items()}

def weighted_fusion(dense: dict, sparse: dict, alpha: float = 0.5, norm=minmax) -> dict[str, float]:
    dn, sn = norm(dense), norm(sparse)

    def floor(nd: dict) -> float:
        # A doc MISSING from a list must rank below everything that list returned.
        # (Using the list's minimum would TIE with its results; with a single result, min-max gives 1.0!)
        if not nd:
            return 0.0
        return min(nd.values()) - 1.0 if norm is zscore else 0.0

    d_missing, s_missing = floor(dn), floor(sn)
    fused = {doc: alpha * dn.get(doc, d_missing) + (1 - alpha) * sn.get(doc, s_missing)
             for doc in set(dn) | set(sn)}
    return dict(sorted(fused.items(), key=lambda x: (-x[1], x[0])))     # deterministic tie-break

def comb_mnz(dense: dict, sparse: dict) -> dict[str, float]:
    dn, sn = minmax(dense), minmax(sparse)
    fused = {doc: (dn.get(doc, 0) + sn.get(doc, 0)) * ((doc in dn) + (doc in sn)) for doc in set(dn) | set(sn)}
    return dict(sorted(fused.items(), key=lambda x: (-x[1], x[0])))

# Toy check: the worked example from Section 4
toy_dense, toy_sparse = ["A", "B", "D"], ["C", "B", "X", "Y", "D"]
print("RRF toy:", {d: round(v, 4) for d, v in rrf([toy_dense, toy_sparse]).items()})

# --- Snippet 4 ---
def hybrid_search(q: str, method: str = "rrf", k: int = 10, n: int = 10,
                  alpha: float = 0.5, rrf_k: int = 60) -> list[str]:
    d, s = dense_scores(q, n), sparse_scores(q, n)
    if method == "dense":
        fused = d
    elif method == "sparse":
        fused = s
    elif method == "rrf":
        fused = rrf([list(d), list(s)], k=rrf_k)
    elif method == "minmax":
        fused = weighted_fusion(d, s, alpha, minmax)
    elif method == "zscore":
        fused = weighted_fusion(d, s, alpha, zscore)
    elif method == "combmnz":
        fused = comb_mnz(d, s)
    else:
        raise ValueError(method)
    return list(fused)[:k]

for method in ("sparse", "dense", "rrf", "minmax", "zscore", "combmnz"):
    print_eval(method, evaluate(lambda q, m=method: hybrid_search(q, m)))

# --- Snippet 5 ---
alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
print(f"{'alpha':>5} {'exact':>6} {'paraph':>6} {'mixed':>6} {'ALL MRR':>8}")
for a in alphas:
    res = evaluate(lambda q, a=a: hybrid_search(q, "minmax", alpha=a))
    print(f"{a:>5} {res['exact']['MRR']:>6.2f} {res['paraphrase']['MRR']:>6.2f} "
          f"{res['mixed']['MRR']:>6.2f} {res['ALL']['MRR']:>8.2f}")

# Simple held-out check: tune on even-indexed queries, test on odd-indexed ones
train, test = ALL_QUERIES[::2], ALL_QUERIES[1::2]
best_a = max(alphas, key=lambda a: evaluate(lambda q: hybrid_search(q, "minmax", alpha=a), train)["ALL"]["MRR"])
test_mrr = evaluate(lambda q: hybrid_search(q, "minmax", alpha=best_a), test)["ALL"]["MRR"]
rrf_test = evaluate(lambda q: hybrid_search(q, "rrf"), test)["ALL"]["MRR"]
print(f"\nBest α on train = {best_a} -> test MRR {test_mrr:.2f}   |   RRF (no tuning) test MRR {rrf_test:.2f}")

# --- Snippet 6 ---
import re
CODE_LIKE = re.compile(r"\b(?=[A-Za-z]*\d)[A-Za-z0-9-]{3,}\b|\b[A-Z]{2,}[A-Z0-9-]*\b")   # E1043, 80C, ZYN-HL-FLEX, IFSC
PROPER_NOUNS = re.compile(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-z]{2,}\b")                          # Radhika, Iyer, Pune

def adaptive_alpha(q: str) -> float:
    signals = len(CODE_LIKE.findall(q)) + len(PROPER_NOUNS.findall(q))
    return 0.3 if signals >= 1 else 0.7       # codes/names -> lean sparse; natural language -> lean dense

for q, *_ in ALL_QUERIES[:3] + ALL_QUERIES[8:10] + ALL_QUERIES[-2:]:
    print(f"α={adaptive_alpha(q)}  {q}")

print()
print_eval("minmax α=0.5", evaluate(lambda q: hybrid_search(q, "minmax", alpha=0.5)))
print_eval("minmax adaptive", evaluate(lambda q: hybrid_search(q, "minmax", alpha=adaptive_alpha(q))))
print_eval("rrf", evaluate(lambda q: hybrid_search(q, "rrf")))

# --- Snippet 7 ---
print("RRF k sensitivity (N=10):")
for rk in (1, 10, 60, 100):
    print_eval(f"  rrf k={rk}", evaluate(lambda q, rk=rk: hybrid_search(q, "rrf", rrf_k=rk)))

print("\nCandidate depth N (RRF k=60):")
for n in (1, 3, 5, 10, 16):
    print_eval(f"  N={n}", evaluate(lambda q, n=n: hybrid_search(q, "rrf", n=n)))

# --- Snippet 8 ---
# Add an INTERNAL document that mentions a public code
INTERNAL = ("RISK-NOTE", "Internal Credit Note",
            "Internal only: branch IFSC ZYNT0004521 is under an elevated fraud-alert review for Q3.")
SEC_DOCS = DOCS + [INTERNAL]
SEC_IDS = [d[0] for d in SEC_DOCS]
SEC_TEXT = [f"{t}. {x}" for _, t, x in SEC_DOCS]
ACCESS = {doc_id: ("internal" if doc_id == "RISK-NOTE" else "public") for doc_id in SEC_IDS}

sec_bm25 = BM25([analyze(t) for t in SEC_TEXT])
sec_vecs = embed(SEC_TEXT)

def allowed_ids(user_levels: set[str]) -> set[str]:
    return {d for d in SEC_IDS if ACCESS[d] in user_levels}

def sec_dense(q, n, allow):
    s = sec_vecs @ embed([q])[0]
    return [SEC_IDS[i] for i in np.argsort(-s) if SEC_IDS[i] in allow][:n]

def sec_sparse(q, n, allow=None):
    hits = [SEC_IDS[i] for i, _ in sec_bm25.search(analyze(q), k=len(SEC_IDS))]
    return [h for h in hits if allow is None or h in allow][:n]

def buggy_hybrid(q, user_levels, n=5):
    allow = allowed_ids(user_levels)
    return list(rrf([sec_dense(q, n, allow), sec_sparse(q, n)]))[:3]    # ❌ sparse branch unfiltered

def secure_hybrid(q, user_levels, n=5):
    allow = allowed_ids(user_levels)
    results = list(rrf([sec_dense(q, n, allow), sec_sparse(q, n, allow)]))[:3]  # ✅ same filter on both
    assert all(r in allow for r in results), "security violation"                 # defense in depth
    return results

q = "Tell me about IFSC ZYNT0004521"
print("Customer, BUGGY  :", buggy_hybrid(q, {"public"}))
print("Customer, SECURE :", secure_hybrid(q, {"public"}))
print("Employee, SECURE :", secure_hybrid(q, {"public", "internal"}))

# --- Snippet 9 ---
from qdrant_client import models

results = qc.query_points(
    collection_name="bank_docs",
    prefetch=[
        models.Prefetch(query=dense_query_vector, using="dense", limit=20),
        models.Prefetch(query=models.SparseVector(indices=sparse_idx, values=sparse_vals), using="sparse", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),        # or models.Fusion.DBSF
    query_filter=security_filter,                             # applied to the whole query
    limit=5,
)

# --- Snippet 10 ---
from weaviate.classes.query import HybridFusion

res = collection.query.hybrid(query="Can I pull surplus money out of ZYN-HL-FLEX?",
                              alpha=0.5, fusion_type=HybridFusion.RELATIVE_SCORE, limit=5)

# --- Snippet 11 ---
from llama_index.core.retrievers import QueryFusionRetriever

hybrid = QueryFusionRetriever(
    [vector_index.as_retriever(similarity_top_k=20), bm25_retriever],    # bm25 from llama-index-retrievers-bm25
    similarity_top_k=5, num_queries=1,                                    # num_queries=1 -> no LLM query rewriting
    mode="reciprocal_rerank", use_async=False,
)
nodes = hybrid.retrieve("Can I pull surplus money out of ZYN-HL-FLEX?")

if __name__ == "__main__":
    print("Starting Day 13 script execution...")
