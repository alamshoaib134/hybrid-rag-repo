# Day 06: Vector Databases (FAISS, Chroma, Pinecone, pgvector)

**Chapter**: Foundations  
**Concept**: The professional warehouse for embeddings: storage, filtering, and retrieval

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this | 
 | ----- | ----- | 
| **Vector DB** | Stores **vectors + text + metadata** and finds nearest neighbors **fast**, with **filters**, **persistence**, and **CRUD**. | 
| **Record** | `id` + `vector` + `document text` + `metadata` (source, date, department, access level...). | 
| **Four families** | **Library** (FAISS) • **Vector DB** (Chroma, Qdrant, Weaviate, Milvus, Pinecone) • **DB extension** (pgvector, MongoDB Atlas, Redis) • **Search engine** (Elasticsearch/OpenSearch). | 
| **FAISS** | Blazing-fast **in-memory library**. Stores vectors only: **you** manage the text, metadata, and filtering. | 
| **Chroma** | Easiest **open-source DB** for prototyping. Embedded or client-server, metadata `where` filters. | 
| **pgvector** | Vectors **inside PostgreSQL**: SQL filters and joins, ACID, backups, and security you already have. | 
| **Pinecone** | **Fully managed** serverless cloud DB, with namespaces, filters, and hybrid search. Zero ops, but your data sits in their cloud. | 
| **Filtering** | **Pre-filter** (filter, then search) keeps K results. **Post-filter** (search, then filter) can return **fewer than K**. | 
| **Updates** | Use **deterministic IDs** (`doc_id#chunk_n`). On a doc change → **delete by `doc_id`**, then re-add. | 
| **Pick one** | Learning → Chroma/FAISS. Already on Postgres → pgvector. No-ops cloud → Pinecone/managed Qdrant. Massive scale → Milvus/Qdrant/Weaviate. |

---
## 📂 Contents & Structure

- [`day_06.ipynb`](./day_06.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_06_vector_databases/day_06.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_06_vector_databases/main.py
```
