# Day 12: Keyword Search (BM25 and Sparse Retrieval)

**Chapter**: Search & Context  
**Concept**: Classic text search vs. Semantic search

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this |
|---|---|
| **Why keywords still matter** | Dense embeddings capture **gist**. Keywords catch **exact tokens**: IDs, codes, names, acronyms, rare terms, new jargon. |
| **Sparse vector** | One dimension per vocabulary term, mostly zeros. Stored as an **inverted index**: term $\rightarrow$ list of docs. |
| **BM25** | `$\sum \text{IDF(term)} \times \text{saturated TF} \times \text{length normalization}$`. Rare terms count more; repeats have **diminishing returns**; long docs are penalized. |
| **Knobs** | `k1` $\approx$ 1.2-2.0 (TF saturation), `b` $\approx$ 0.75 (length normalization). The defaults work well. |
| **Analyzer** | Tokenize $\rightarrow$ lowercase $\rightarrow$ stopwords $\rightarrow$ stemming. **Protect codes** (`ZYN-HL-FLEX`, `E1043`, `80C`). Use the **same** analyzer for docs and queries. |
| **Weakness** | **Vocabulary mismatch**: "house" $\neq$ "home loan", "freeze" $\neq$ "block". No synonyms or meaning. |
| **Learned sparse** | SPLADE / ELSER / BGE-M3 sparse: a model **adds related terms** and weights them, keeping the inverted-index speed. |
| **Tools** | `rank_bm25`, `bm25s`, SQLite **FTS5** (built into Python), Elasticsearch/OpenSearch, Pyserini/Lucene, Tantivy, built-in BM25 in Weaviate/Qdrant/ParadeDB. |
| **Takeaway** | Sparse and dense **fail differently**, so **combine them** (hybrid search, Day 13). |

---

---
## 📂 Contents & Structure

- [`day_12.ipynb`](./day_12.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_12_keyword_search_bm25/day_12.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_12_keyword_search_bm25/main.py
```
