"""
Day 14: Re-ranking with Cross-Encoders and Cohere Rerank
Core Concept: Sorting retrieved results for maximum accuracy with cross-encoders
Chapter: Search & Context

This script executes the key algorithmic implementations and examples from Day 14.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 14: Re-ranking with Cross-Encoders and Cohere Rerank")
    print("=" * 60)

# --- Snippet 1 ---
pairs = [(query, candidate["text"]) for candidate in candidates]
scores = model.predict(pairs, batch_size=32)

# --- Snippet 2 ---
{
    "id": "policy-2026#chunk-04",
    "text": "Early repayment has no fee after 12 months...",
    "metadata": {"tenant_id": "acme", "effective_year": 2026},
    "dense_score": 0.71,
    "bm25_score": 8.4,
    "rrf_score": 0.0318,
    "rerank_score": 0.93,
    "original_rank": 7,
    "rerank_rank": 1,
}

# --- Snippet 3 ---
import math
import re
import time
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9$.-]+")

def tokens(text):
    return TOKEN_RE.findall(text.lower())

def features(query, document):
    """Transparent pair features; this is not a neural cross-encoder."""
    q = tokens(query)
    d = tokens(document)
    qs, ds = set(q), set(d)
    overlap = len(qs & ds) / max(1, len(qs))

    q_numbers = {t for t in qs if any(c.isdigit() for c in t)}
    number_match = len(q_numbers & ds) / max(1, len(q_numbers)) if q_numbers else 1.0

    negations = {"no", "not", "never", "without"}
    q_neg = bool(qs & negations)
    d_neg = bool(ds & negations)
    negation_agreement = 1.0 if q_neg == d_neg else 0.0

    phrase = 1.0 if " ".join(q) in " ".join(d) else 0.0
    return {
        "overlap": overlap,
        "number_match": number_match,
        "negation_agreement": negation_agreement,
        "phrase": phrase,
    }

def offline_pair_score(query, document):
    f = features(query, document)
    return (
        0.55 * f["overlap"]
        + 0.20 * f["number_match"]
        + 0.20 * f["negation_agreement"]
        + 0.05 * f["phrase"]
    )

def rerank(query, candidates, score_fn=offline_pair_score, top_n=None):
    scored = []
    for original_rank, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        item["original_rank"] = original_rank
        item["rerank_score"] = float(score_fn(query, item["text"]))
        scored.append(item)

    scored.sort(key=lambda x: (-x["rerank_score"], x["original_rank"], x["id"]))
    for rerank_rank, item in enumerate(scored, start=1):
        item["rerank_rank"] = rerank_rank
    return scored[:top_n] if top_n is not None else scored

if __name__ == "__main__":
    query = "Which loan has no prepayment fee after 12 months?"
    candidates = [
        {
            "id": "d1",
            "text": "Flex Loan charges a 2% prepayment fee after 12 months.",
            "rrf_score": 0.033,
        },
        {
            "id": "d2",
            "text": "Prime Loan has no prepayment fee after 12 months.",
            "rrf_score": 0.031,
        },
        {
            "id": "d3",
            "text": "Flex Loan has no late payment fee during the first month.",
            "rrf_score": 0.029,
        },
        {
            "id": "d4",
            "text": "Prime Loan rates are reviewed every 12 months.",
            "rrf_score": 0.026,
        },
    ]

    ranked = rerank(query, candidates, top_n=3)
    for item in ranked:
        print(
            item["rerank_rank"],
            item["id"],
            round(item["rerank_score"], 3),
            "was", item["original_rank"],
        )

# --- Snippet 4 ---
def precision_at_k(ranked_ids, relevant_ids, k):
    selected = ranked_ids[:k]
    return sum(doc_id in relevant_ids for doc_id in selected) / k

def reciprocal_rank(ranked_ids, relevant_ids):
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def dcg_at_k(ranked_ids, grades, k):
    return sum(
        (2 ** grades.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )

def ndcg_at_k(ranked_ids, grades, k):
    actual = dcg_at_k(ranked_ids, grades, k)
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    ideal = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return actual / ideal if ideal else 0.0

before = ["d1", "d2", "d3", "d4"]
after = [item["id"] for item in ranked]
relevant = {"d2"}
grades = {"d1": 0, "d2": 3, "d3": 0, "d4": 0}

print("MRR before/after:", reciprocal_rank(before, relevant), reciprocal_rank(after, relevant))
print("nDCG@3 before/after:", ndcg_at_k(before, grades, 3), ndcg_at_k(after, grades, 3))

# --- Snippet 5 ---
def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 1.0
    return len(set(ranked_ids[:k]) & set(relevant_ids)) / len(relevant_ids)

all_candidates = [
    {"id": "wrong-1", "text": "General loan fees and rates."},
    {"id": "wrong-2", "text": "How monthly repayments are calculated."},
    {"id": "answer", "text": "Prime Loan has no prepayment fee after 12 months."},
]

for candidate_depth in (1, 2, 3):
    shortlist = all_candidates[:candidate_depth]
    output = rerank(query, shortlist)
    ids = [item["id"] for item in output]
    print(
        "depth", candidate_depth,
        "candidate_recall", recall_at_k(ids, {"answer"}, candidate_depth),
        "top1", ids[0],
    )

# --- Snippet 6 ---
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cross_encoder_rerank(query, candidates, top_n=5):
    pairs = [(query, item["text"]) for item in candidates]
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)

    output = []
    for original_rank, (candidate, score) in enumerate(
        zip(candidates, scores), start=1
    ):
        item = dict(candidate)
        item["original_rank"] = original_rank
        item["rerank_score"] = float(score)
        output.append(item)

    output.sort(key=lambda x: (-x["rerank_score"], x["original_rank"], x["id"]))
    return output[:top_n]

# --- Snippet 7 ---
import os
import cohere

def cohere_rerank(query, candidates, top_n=5):
    api_key = os.environ["COHERE_API_KEY"]
    client = cohere.ClientV2(api_key=api_key)
    response = client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=[item["text"] for item in candidates],
        top_n=min(top_n, len(candidates)),
    )

    output = []
    for rerank_rank, result in enumerate(response.results, start=1):
        original = candidates[result.index]
        item = dict(original)
        item["original_rank"] = result.index + 1
        item["rerank_rank"] = rerank_rank
        item["rerank_score"] = float(result.relevance_score)
        output.append(item)
    return output

# --- Snippet 8 ---
def benchmark_depths(query, candidates, relevant_ids, depths, repeats=200):
    rows = []
    for depth in depths:
        subset = candidates[:depth]
        start = time.perf_counter()
        for _ in range(repeats):
            result = rerank(query, subset)
        elapsed_ms = (time.perf_counter() - start) * 1000 / repeats
        ids = [item["id"] for item in result]
        rows.append(
            {
                "depth": depth,
                "candidate_recall": recall_at_k(
                    [item["id"] for item in subset], relevant_ids, depth
                ),
                "mrr": reciprocal_rank(ids, relevant_ids),
                "ms": elapsed_ms,
            }
        )
    return rows

for row in benchmark_depths(
    query, all_candidates, {"answer"}, depths=[1, 2, 3]
):
    print(row)

# --- Snippet 9 ---
def retrieve_then_rerank(query, identity, retriever, reranker, top_n=5):
    # The retriever must enforce these constraints at the data store.
    filters = {
        "tenant_id": identity["tenant_id"],
        "access_groups": {"$in": identity["groups"]},
        "is_active": True,
    }
    candidates = retriever.search(query=query, top_k=50, filters=filters)

    for item in candidates:
        if item["metadata"]["tenant_id"] != identity["tenant_id"]:
            raise PermissionError("Retriever returned a cross-tenant candidate")

    return reranker(query, candidates, top_n=top_n)

# --- Snippet 10 ---
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5,
)

query_engine = index.as_query_engine(
    similarity_top_k=50,
    node_postprocessors=[reranker],
)

if __name__ == "__main__":
    print("Starting Day 14 script execution...")
