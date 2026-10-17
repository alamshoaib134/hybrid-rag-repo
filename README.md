# Hybrid RAG Systems — Code & Interactive Curriculum

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, code-first engineering curriculum and repository for building **Production-Grade Hybrid Retrieval-Augmented Generation (RAG) Systems**.

This repository contains **15 complete days** of interactive Jupyter notebooks, standalone python scripts, and in-depth conceptual explanations — taking you from fundamental token mechanics to state-of-the-art hybrid search (Dense + Sparse), Reciprocal Rank Fusion (RRF), Cross-Encoder Re-ranking, and dynamic Query Transformations.

---

## 🏗️ Modern Hybrid RAG Architecture

```
                   +-------------------------------------------------------------+
                   |                     OFFLINE INGESTION                       |
                   |                                                             |
                   |   Documents (PDF/HTML/MD) -> Clean & Normalize -> Chunk     |
                   |                                       |                     |
                   |           +---------------------------+-----------+         |
                   |           |                                       |         |
                   |    Dense Embedder                          Sparse Index     |
                   | (SentenceTransformers)                       (BM25)         |
                   |           |                                       |         |
                   |           v                                       v         |
                   |     Vector Database                        Inverted Index   |
                   |   (Chroma / FAISS / HNSW)                    (BM25 Store)   |
                   +-------------------------------------------------------------+
                                                       |
+-------------------------------------------------------------------------------------------------------+
|                                           ONLINE RETRIEVAL                                            |
|                                                                                                       |
|  User Query ----> [ Query Transformation: HyDE / Multi-Query / Sub-Questions / Rewriting ]            |
|                                           |                                                           |
|                     +---------------------+---------------------+                                     |
|                     |                                           |                                     |
|                     v                                           v                                     |
|        [ Dense Semantic Search ]                    [ Sparse Lexical Search ]                         |
|           (Cosine / Dot / L2)                                (BM25)                                   |
|                     |                                           |                                     |
|                     +---------------------+---------------------+                                     |
|                                           |                                                           |
|                                           v                                                           |
|                       [ Reciprocal Rank Fusion (RRF) / Hybrid Blending ]                              |
|                                           |                                                           |
|                                           v                                                           |
|                         [ Cross-Encoder Re-ranking (Top K) ]                                          |
|                                           |                                                           |
|                                           v                                                           |
|                       [ Prompt Assembly & Context Compression ]                                       |
|                                           |                                                           |
|                                           v                                                           |
|                       [ Grounded LLM Generation & Source Citations ]                                  |
+-------------------------------------------------------------------------------------------------------+
```

---

## 📚 15-Day Course Curriculum Matrix

| Day | Topic | Key Concepts & Mechanics | Notebook | Script |
|:---:|:------|:-------------------------|:--------:|:------:|
| **01** | **LLM Basics** | Tokens, Tiktoken, Context Windows, Temperature/Top-p, Prompt Engineering (Zero/Few-Shot/CoT) | [`day_01.ipynb`](./day_01_llm_basics/day_01.ipynb) | [`main.py`](./day_01_llm_basics/main.py) |
| **02** | **Hallucinations & RAG** | Factuality failure modes, Grounded Generation, Self-Consistency, LLM-as-a-Judge Faithfulness Evaluator | [`day_02.ipynb`](./day_02_hallucinations_and_rag/day_02.ipynb) | [`main.py`](./day_02_hallucinations_and_rag/main.py) |
| **03** | **RAG Architecture From Scratch** | Complete pure-Python pipeline: Chunker, Embedder, Memory Store, Retriever, Generator, Recall@K & MRR | [`day_03.ipynb`](./day_03_rag_architecture/day_03.ipynb) | [`main.py`](./day_03_rag_architecture/main.py) |
| **04** | **Text Embeddings** | Vector geometries, Word2Vec vs. Transformers, `sentence-transformers`, OpenAI embeddings, Pooling | [`day_04.ipynb`](./day_04_text_embeddings/day_04.ipynb) | [`main.py`](./day_04_text_embeddings/main.py) |
| **05** | **Similarity Metrics** | Cosine Similarity, Dot Product, Euclidean Distance (L2), Manhattan (L1), Unit Normalization Equivalence | [`day_05.ipynb`](./day_05_similarity_metrics/day_05.ipynb) | [`main.py`](./day_05_similarity_metrics/main.py) |
| **06** | **Vector Databases** | ChromaDB & FAISS in depth: Collections, Embeddings Ingestion, Metadata Filtering, CRUD & Persistence | [`day_06.ipynb`](./day_06_vector_databases/day_06.ipynb) | [`main.py`](./day_06_vector_databases/main.py) |
| **07** | **Basic RAG Pipeline** | Production Frameworks: LangChain LCEL Runnables, LlamaIndex QueryEngine, Source Nodes, MMR Retrieval | [`day_07.ipynb`](./day_07_basic_rag_pipeline/day_07.ipynb) | [`main.py`](./day_07_basic_rag_pipeline/main.py) |
| **08** | **Document Loaders** | Parsing complex documents: PDF text & table extraction (`pdfplumber`, `pymupdf4llm`), HTML, Markdown | [`day_08.ipynb`](./day_08_document_loaders/day_08.ipynb) | [`main.py`](./day_08_document_loaders/main.py) |
| **09** | **Chunking Strategies** | Fixed character, Recursive text splitting, Semantic chunking (embedding distance shifts), Overlap tuning | [`day_09.ipynb`](./day_09_chunking_strategies/day_09.ipynb) | [`main.py`](./day_09_chunking_strategies/main.py) |
| **10** | **Metadata & Filtering** | Structured tagging, Pre-filtering vs Post-filtering benchmarks, Self-Querying retrieval simulation | [`day_10.ipynb`](./day_10_metadata_and_filtering/day_10.ipynb) | [`main.py`](./day_10_metadata_and_filtering/main.py) |
| **11** | **Indexing Methods** | ANN Indexing: Flat Brute Force vs. IVF (Inverted File) vs. HNSW (Navigable Small World) vs. Quantization (PQ) | [`day_11.ipynb`](./day_11_indexing_methods/day_11.ipynb) | [`main.py`](./day_11_indexing_methods/main.py) |
| **12** | **Keyword Search (BM25)** | Inverted Indexes, Lexical Search, TF-IDF vs BM25Okapi, Length Normalization, Term Saturation | [`day_12.ipynb`](./day_12_keyword_search_bm25/day_12.ipynb) | [`main.py`](./day_12_keyword_search_bm25/main.py) |
| **13** | **Hybrid Search (Dense + Sparse)** | Reciprocal Rank Fusion (RRF), Linear Score Normalization, Hybrid Retriever combining FAISS/Chroma + BM25 | [`day_13.ipynb`](./day_13_hybrid_search/day_13.ipynb) | [`main.py`](./day_13_hybrid_search/main.py) |
| **14** | **Re-ranking with Cross-Encoders** | Two-Stage Retrieval: Bi-Encoder Candidate Generation + Cross-Encoder Joint Attention Scoring, Cohere Rerank | [`day_14.ipynb`](./day_14_reranking_cross_encoders/day_14.ipynb) | [`main.py`](./day_14_reranking_cross_encoders/main.py) |
| **15** | **Query Transformation** | Advanced Retrieval Techniques: Query Rewriting, Sub-Query Decomposition, Multi-Query Expansion, HyDE, Routing | [`day_15.ipynb`](./day_15_query_transformation/day_15.ipynb) | [`main.py`](./day_15_query_transformation/main.py) |

---

## ⚡ Quickstart Guide

### 1. Clone or Open the Repository
```bash
cd /Users/shoaib/Desktop/hybrid-rag-repo
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert any required API keys (e.g. OpenAI, Cohere):
```bash
cp .env.example .env
```
*(Note: Most days use local SentenceTransformers, BM25, and NumPy implementations which run 100% offline without any API keys!)*

### 5. Verify Your Environment
```bash
python verify_setup.py
```

### 6. Launch Jupyter
```bash
jupyter notebook
```
Navigate to any day's folder (e.g., `day_01_llm_basics/day_01.ipynb`) and run the cells interactively!

---

## 💡 How Each Day Is Organized

Every day module is structured identically for consistency:
```
day_XX_topic_name/
├── day_XX.ipynb      # Complete interactive notebook with math, architecture, and executable code
├── main.py           # Standalone Python script with runnable algorithms and demonstrations
└── README.md         # Fast-reference cheat sheet, formula breakdown, and usage instructions
```

---

## 🤝 Prerequisites & Philosophy
- **Python**: Intermediate Python (dataclasses, list comprehensions, type hinting).
- **No Magic**: Whenever possible, concepts are built from first principles (NumPy, pure Python) before demonstrating high-level frameworks (LangChain, LlamaIndex, ChromaDB).
- **Production-Ready**: Focuses on evaluation, failure modes, latency, memory footprint, and edge cases.
