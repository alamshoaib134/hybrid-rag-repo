# Day 13: Hybrid Search (Dense + Sparse)

**Chapter**: Search & Context  
**Concept**: Combining vector search and keyword search with Reciprocal Rank Fusion

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this | 
| ----- | ----- | 
| **Why hybrid?** | Dense catches **meaning**; sparse catches **exact tokens**. They fail on **different** queries, so combine them. | 
| **Standard pattern** | Retrieve top-N from **each** retriever (for example, 20-100) -> **fuse** -> top-K (-> re-rank, Day 14). | 
| **The core problem** | Scores live on **different scales** (BM25 is unbounded; cosine is roughly -1..1). Never add raw scores. | 
| **RRF** | `score = Σ 1 / (k + rank)`, with `k ≈ 60`. **Rank-based**, scale-free, no tuning. **The robust default.** | 
| **Weighted fusion** | Normalize (min-max / z-score) -> `α·dense + (1-α)·sparse`. Tune `α` on an eval set. | 
| **α meaning** | `α = 1` -> pure dense; `α = 0` -> pure sparse; `0.5-0.7` is a typical start. | 
| **Adaptive** | Queries containing **codes/IDs** -> lean sparse; natural-language questions -> lean dense. | 
| **Security** | Apply the **same filters to both branches**. A missed filter on one branch **leaks** data. | 
| **DB support** | Elasticsearch/OpenSearch, Weaviate, Qdrant, Milvus, Pinecone, Vespa, Azure AI Search, and SQL (pgvector + full-text) all support hybrid. |

---
## 📂 Contents & Structure

- [`day_13.ipynb`](./day_13.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_13_hybrid_search/day_13.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_13_hybrid_search/main.py
```
