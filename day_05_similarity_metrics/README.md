# Day 05: Similarity Metrics (Cosine, Dot Product, Euclidean)

**Chapter**: Foundations  
**Concept**: The geometry of relevance: Cosine, Dot Product, and Euclidean

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this |
|---|---|
| **Dot product** | $\sum a_i b_i$. Higher = better. Uses **direction + length**. |
| **Cosine** | $\text{dot} / (\Vert a \Vert \Vert b \Vert)$. Higher = better. Uses **direction only**. Range $[-1, 1]$. |
| **Euclidean (L2)** | $\Vert a - b \Vert$. **Lower = better** (a distance). |
| **Key identity** | For unit vectors: $\cos = \text{dot}$, and $L2^2 = 2 - 2 \cdot \cos$, so **all rankings are identical**. |
| **Best practice** | **Normalize + inner product** = cosine quality at dot-product speed. |
| **Which metric?** | The one the **model was trained with** (check the model card). |
| **Distances** | Lower = better. Cosine distance = $1 - \cos$. FAISS `IndexFlatL2` returns **squared** L2. Chroma defaults to **L2**. |
| **Thresholds** | **Model-specific.** Tune on labeled answerable/unanswerable questions. |
| **Brute force** | $O(N \cdot d)$ per query. Fine up to ~100k-1M vectors; beyond that, use **ANN** (Day 11). |
| **MMR** | $\lambda = 1 \rightarrow$ pure relevance, $\lambda = 0 \rightarrow$ pure diversity. Typical **0.5-0.7**. |

---

---
## 📂 Contents & Structure

- [`day_05.ipynb`](./day_05.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_05_similarity_metrics/day_05.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_05_similarity_metrics/main.py
```
