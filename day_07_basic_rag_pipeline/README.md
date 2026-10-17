# Day 07: Build a Basic RAG Pipeline (LangChain and LlamaIndex)

**Chapter**: Foundations  
**Concept**: Assembling the pieces with LangChain and LlamaIndex

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this |
|---|---|
| **Why a framework?** | Ready-made **loaders, splitters, embedders, vector stores, retrievers, prompts, LLM wrappers**, all swappable behind common interfaces. |
| **LangChain** | A general **LLM app toolkit**. Build pipelines by composing **Runnables** with `\|` (LCEL). |
| **LlamaIndex** | A **data/RAG-first** framework: `Documents -> Nodes -> Index -> QueryEngine`, with sensible defaults. |
| **LangChain RAG in 4 lines** | `splitter.split_documents()` -> `Chroma.from_documents()` -> `.as_retriever()` -> `{context, question} \| prompt \| llm \| parser`. |
| **LlamaIndex RAG in 3 lines** | `VectorStoreIndex.from_documents(docs)` -> `.as_query_engine()` -> `.query("...")`. |
| **Sources** | LangChain: return retrieved docs alongside the answer (`RunnableParallel`). LlamaIndex: `response.source_nodes`. |
| **Retriever options** | `similarity`, `mmr` (diversity), `similarity_score_threshold`, plus **metadata filters**. |
| **Watch out** | Hidden defaults (LlamaIndex defaults to OpenAI; Chroma defaults to L2), **chunk size in characters vs tokens**, fast-changing imports. |
| **Golden rule** | Frameworks change **how much code you write**, not **how RAG works**. Evaluate exactly as before. |

---

---
## 📂 Contents & Structure

- [`day_07.ipynb`](./day_07.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_07_basic_rag_pipeline/day_07.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_07_basic_rag_pipeline/main.py
```
