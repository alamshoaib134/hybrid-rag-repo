# Day 15: Query Transformation (Rewriting, Decomposition, Multi-Query, HyDE, Routing)

**Chapter**: Search & Context  
**Concept**: Rewriting and routing user queries for superior retrieval

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this | 
| ----- | ----- | 
| **The problem** | Weeks 1-2 tuned the **index**. The **query** is still whatever the user typed: vague, misspelled, conversational, or multi-part. | 
| **Where it runs** | Before retrieval: `query -> transform -> retrieve -> fuse -> re-rank -> generate`. | 
| **Rewriting** | Normalize, expand acronyms and synonyms, resolve pronouns from chat history. Cheapest and highest hit rate. | 
| **Multi-query (RAG-Fusion)** | Generate 3-5 paraphrases, retrieve for each, fuse with **RRF** (Day 13). Fixes one-shot phrasing luck. | 
| **Decomposition** | Split compound and multi-hop questions into sub-questions. Measure **coverage**, not just recall of one doc. | 
| **HyDE** | Embed a *hypothetical answer* instead of the question, so you compare answer-to-answer. Helps on jargon-poor queries, hurts on exact codes. | 
| **Step-back** | Ask the broader question first to pull in background context, then the specific one. | 
| **Routing** | Choose index, filter, retriever mix, or **no retrieval at all**. Always define a fallback route. | 
| **Query drift** | The transform changes the meaning and retrieval gets *worse*. **Always keep the original query in the fan-out.** | 
| **Budget** | Every transform adds an LLM call and more retrievals. Cap fan-out, run branches concurrently, cache aggressively. | 
| **Security** | A rewritten query must **never** build filters, tenant IDs, or ACLs. Treat query text as untrusted input. | 
| **Default start** | Rule-based normalization + conversational rewrite + original query always included. Add multi-query only if evaluation justifies it. |

---
## 📂 Contents & Structure

- [`day_15.ipynb`](./day_15.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_15_query_transformation/day_15.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_15_query_transformation/main.py
```
