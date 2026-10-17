# Day 10: Metadata and Filtering

**Chapter**: Search & Context  
**Concept**: Adding tags to chunks for precise searching

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this |
|---|---|
| **Division of labor** | **Similarity** decides *what's relevant*. **Metadata filters** decide *what's allowed and valid*. |
| **Schema** | Identity + structure + time + business scope + provenance. Normalize values; validate at ingestion. |
| **Types** | Store dates as **sortable numbers** (`20260401`) or timestamps. Use consistent lowercase codes (`home_loan`). |
| **Freshness** | **Effective dating**: `effective_from $\le$ as_of < effective_to`. Supports "current" and "as of 2023" questions. |
| **Access control** | Build the filter from the **authenticated user**, **never** from the prompt. Re-check results. Audit. |
| **Self-query** | Extract filters from the question (rules or LLM) $\rightarrow$ **whitelist and validate** $\rightarrow$ relax if zero results. |
| **Hard vs soft** | Hard filter = exclude. Soft signal = **boost** (recency, authority) inside the ranking. |
| **Pitfalls** | Missing fields silently excluded $\cdot$ inconsistent values $\cdot$ string dates $\cdot$ over-filtering $\rightarrow$ zero results $\cdot$ hallucinated LLM filters. |

---

---
## 📂 Contents & Structure

- [`day_10.ipynb`](./day_10.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_10_metadata_and_filtering/day_10.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_10_metadata_and_filtering/main.py
```
