# Day 14: Re-ranking with Cross-Encoders and Cohere Rerank

**Chapter**: Search & Context  
**Concept**: Sorting retrieved results for maximum accuracy with cross-encoders

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this |
|---|---|
| **Purpose** | Retrieval gets a high-recall shortlist; re-ranking puts the truly useful chunks first. |
| **Standard pipeline** | Dense + BM25 -> fuse top **20-100** -> re-rank -> keep top **3-10** -> LLM. |
| **Bi-encoder** | Encodes query and document separately. Fast and indexable, but interaction is compressed into one vector similarity. |
| **Cross-encoder** | Reads `[query, document]` together and outputs one relevance score. Slower, but sees token-level interactions. |
| **Main rule** | A re-ranker can reorder candidates; it **cannot recover a relevant document that retrieval missed**. |
| **Good at** | Negation, exact clauses, dates, numbers, constraints, and distinguishing closely related chunks. |
| **Evaluate** | Candidate Recall@N first; then MRR, nDCG@K, Precision@K, and downstream answer quality. |
| **Latency controls** | Candidate count, document length, model size, batching, hardware, caching, and timeouts. |
| **Security** | Filter unauthorized documents **before** re-ranking. Never send forbidden text to a local or hosted model. |
| **Default start** | Retrieve 50 -> re-rank 50 -> pass top 5; measure quality and p95 latency, then tune. |

---

---
## 📂 Contents & Structure

- [`day_14.ipynb`](./day_14.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_14_reranking_cross_encoders/day_14.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_14_reranking_cross_encoders/main.py
```
