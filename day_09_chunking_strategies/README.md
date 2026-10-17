# Day 09: Chunking Strategies (Fixed, Recursive, Semantic)

**Chapter**: Search & Context  
**Concept**: Breaking documents into optimal context units

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this |
|---|---|
| **Chunk** | The unit of **retrieval** *and* the unit of **context**. It must make sense **on its own**. |
| **Size trade-off** | Too small $\rightarrow$ loses context ("it", "this category"). Too large $\rightarrow$ blurry embeddings, noise, cost. |
| **Starting point** | **200-500 tokens**, **10-20% overlap**, respect structure. Tune with evaluation. |
| **Fixed-size** | Every N tokens. Simple and predictable, but **cuts mid-sentence**. |
| **Sentence-based** | Pack whole sentences up to N tokens. Clean boundaries. |
| **Recursive** | Split on `\n\n` $\rightarrow$ `\n` $\rightarrow$ `.` $\rightarrow$ ` ` until it fits. **The default for most text.** |
| **Structure-aware** | Split by **headings/sections**; keep **tables and lists whole**; prefix the **section path**. Often best for structured docs. |
| **Semantic** | Break where **embedding similarity drops** between sentences. Topic-coherent, but costlier and needs tuning. |
| **Advanced** | Propositions (LLM), contextual chunk headers, late chunking, parent-child / small-to-big (Day 18). |
| **Evaluate** | **Answer-containment Recall@K** + **context tokens per query** + chunk statistics. Let the numbers decide. |

---

---
## 📂 Contents & Structure

- [`day_09.ipynb`](./day_09.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_09_chunking_strategies/day_09.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_09_chunking_strategies/main.py
```
