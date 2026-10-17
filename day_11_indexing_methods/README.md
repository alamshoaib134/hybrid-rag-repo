# Day 11: Indexing Methods (Flat, IVF, HNSW, and Quantization)

**Chapter**: Search & Context  
**Concept**: How vector databases organize data for fast retrieval

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this |
|---|---|
| **Why ANN?** | Exact search costs $\mathcal{O}(N \cdot d)$ per query. At millions of vectors, you trade a little **recall** for a lot of **speed**. |
| **The triangle** | **Recall** $\leftrightarrow$ **latency** $\leftrightarrow$ **memory** (plus build time). You tune for an operating point, not a perfect score. |
| **Flat** | Exact, no training, 100% recall. Perfect up to ~100k-1M vectors. Also your **ground truth**. |
| **IVF** | k-means into `nlist` clusters; search the `nprobe` nearest clusters. Needs **training**. `nlist` $\approx \sqrt{N} - 4\sqrt{N}$; raise `nprobe` for recall. |
| **HNSW** | A multi-layer **graph** with greedy navigation. `M` (links: memory/recall), `efConstruction` (build quality), `efSearch` (query recall/latency). The default in most vector DBs. |
| **Quantization** | **SQ8** (4x smaller), **PQ** (16-64x smaller), **binary** (32x smaller). Recover recall with **re-ranking** (refine) on full vectors. |
| **Measure** | Recall@k vs **exact** results + ms/query + MB + build time. Target something like **recall@10** $\ge 0.95$. |
| **Gotchas** | FAISS IVF `nprobe` defaults to $1$; HNSW `efSearch` defaults to $16$. Selective filters hurt HNSW. ANN recall $\ne$ answer relevance. |

---

---
## 📂 Contents & Structure

- [`day_11.ipynb`](./day_11.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_11_indexing_methods/day_11.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_11_indexing_methods/main.py
```
