# Hybrid RAG Systems: Production-Grade Dense & Sparse Retrieval

<div align="center">

[![AI From Scratch](https://img.shields.io/badge/Platform-AI_From_Scratch-6366f1.svg?style=for-the-badge&logo=google-chrome&logoColor=white)](https://www.aifromscratch.in/)
[![Course: Hybrid RAG](https://img.shields.io/badge/Course-Hybrid_RAG_Systems-059669.svg?style=for-the-badge)](https://www.aifromscratch.in/course/hybrid-rag)
[![Free AI Roadmap](https://img.shields.io/badge/Roadmap-Free_AI_Guide-blue.svg?style=for-the-badge)](https://www.aifromscratch.in/learn-ai-roadmap)
[![Daily Papers](https://img.shields.io/badge/Daily_Papers-Research_Simplified-purple.svg?style=for-the-badge)](https://www.aifromscratch.in/papers)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**An end-to-end, code-first engineering curriculum for building fiduciary-grade, production Hybrid Retrieval-Augmented Generation (RAG) systems.**

[Interactive Course](https://www.aifromscratch.in/course/hybrid-rag) • [AI Learning Roadmap](https://www.aifromscratch.in/learn-ai-roadmap) • [Daily Research Papers](https://www.aifromscratch.in/papers) • [Main Website](https://www.aifromscratch.in/)

</div>

---

> 🎓 **Official Companion Repository for [AI From Scratch](https://www.aifromscratch.in/)**  
> This open-source repository contains the runnable algorithms, mathematical derivations, and 15 interactive Jupyter notebooks for the **[Hybrid RAG Systems Course](https://www.aifromscratch.in/course/hybrid-rag)** hosted on **[AI From Scratch](https://www.aifromscratch.in/)**.
>
> **Explore the full learning ecosystem on AI From Scratch:**
> - 🌐 **Main Academy**: [Learn AI from Scratch](https://www.aifromscratch.in/) — Free beginner guides, step-by-step tutorials, and engineering masterclasses.
> - 🚀 **Interactive RAG Track**: [Hybrid RAG Systems Course](https://www.aifromscratch.in/course/hybrid-rag) — Master Dense + Sparse search, Re-ranking, and dynamic Query Decomposition.
> - 📘 **Beginner 30-Day Track**: [Learn AI for Beginners — 30-Day Guide](https://www.aifromscratch.in/course/30-days-of-ai) — From tokenization fundamentals to AGI concepts.
> - 🗺️ **Career Roadmap**: [Free AI Learning Roadmap](https://www.aifromscratch.in/learn-ai-roadmap) — Step-by-step visual roadmap from Python basics to production LLMOps.
> - 📑 **Daily Frontier Research**: [Daily AI Research Paper Breakdowns](https://www.aifromscratch.in/papers) — Simplified breakdowns of breakthrough machine learning papers.

---

## 📖 Table of Contents

- [Why Hybrid RAG? (Dense vs. Sparse)](#-why-hybrid-rag)
- [Architecture Blueprint](#-modern-hybrid-rag-architecture)
- [15-Day Course Curriculum Matrix](#-15-day-course-curriculum-matrix)
- [Core Mathematical & Algorithmic Foundations](#-core-mathematical--algorithmic-foundations)
- [Quickstart Guide](#-quickstart-guide)
- [Repository Structure](#-repository-structure)
- [Frequently Asked Questions (FAQ)](#-frequently-asked-questions-faq)
- [About the Author & AI From Scratch](#-about-the-author--ai-from-scratch)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Why Hybrid RAG?

Standard vector-only RAG pipelines rely exclusively on **dense embeddings** (e.g., SentenceTransformers, OpenAI `text-embedding-3`). While dense vectors capture abstract semantic intent, they suffer from critical blind spots in enterprise and production environments:

| Retrieval Dimension | Dense Semantic Search (Vectors) | Sparse Lexical Search (BM25) | **Hybrid RAG (Dense + BM25 + Re-Rank)** |
|:---|:---|:---|:---|
| **Exact Keyword Matching** | ⚠️ Poor (often misses exact SKUs, codes, IDs) | ✅ Outstanding (inverted index exact match) | 🏆 **Flawless precision across tokens** |
| **Synonym & Paraphrase** | ✅ High (understands contextual similarity) | ❌ Ineffective (fails on vocabulary mismatch) | 🏆 **Understands meaning and synonyms** |
| **Out-Of-Vocabulary (OOV)** | ⚠️ Compresses unfamiliar tokens into noisy space | ✅ Accurately indexes unique domain terms | 🏆 **Robust against rare terms & typos** |
| **Negative Constraints** | ❌ Struggling (e.g., "without battery") | ⚠️ Partial (term frequency penalization) | 🏆 **Filtered by metadata & Cross-Encoder** |
| **Mathematical / Legal Terms** | ⚠️ Vulnerable to semantic drift | ✅ Preserves precise statutory or formulaic tokens | 🏆 **Fiduciary-grade reliability** |

By fusing **Dense Vector Search** (semantic context) and **Sparse Lexical Search (BM25)** via **Reciprocal Rank Fusion (RRF)**, followed by a second-stage **Cross-Encoder Re-ranking**, Hybrid RAG achieves state-of-the-art precision and recall across complex knowledge bases.

For interactive walkthroughs and visual benchmarks, explore the **[Hybrid RAG Systems course on AI From Scratch](https://www.aifromscratch.in/course/hybrid-rag)**.

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

Each module below contains pure-Python reference implementations, math proofs, and executable Jupyter notebooks. Interactive web lessons and self-assessment quizzes are available on **[AI From Scratch](https://www.aifromscratch.in/)**.

| Day | Topic | Key Concepts & Mechanics | Web Lesson | Notebook | Script |
|:---:|:------|:-------------------------|:----------:|:--------:|:------:|
| **01** | **LLM Basics** | Tokens, Tiktoken, Context Windows, Temperature/Top-p, Prompt Engineering (Zero/Few-Shot/CoT) | [Start Day 01](https://www.aifromscratch.in/course/hybrid-rag) | [`day_01.ipynb`](./day_01_llm_basics/day_01.ipynb) | [`main.py`](./day_01_llm_basics/main.py) |
| **02** | **Hallucinations & RAG** | Factuality failure modes, Grounded Generation, Self-Consistency, LLM-as-a-Judge Faithfulness Evaluator | [Start Day 02](https://www.aifromscratch.in/course/hybrid-rag) | [`day_02.ipynb`](./day_02_hallucinations_and_rag/day_02.ipynb) | [`main.py`](./day_02_hallucinations_and_rag/main.py) |
| **03** | **RAG Architecture From Scratch** | Complete pure-Python pipeline: Chunker, Embedder, Memory Store, Retriever, Generator, Recall@K & MRR | [Start Day 03](https://www.aifromscratch.in/course/hybrid-rag) | [`day_03.ipynb`](./day_03_rag_architecture/day_03.ipynb) | [`main.py`](./day_03_rag_architecture/main.py) |
| **04** | **Text Embeddings** | Vector geometries, Word2Vec vs. Transformers, `sentence-transformers`, OpenAI embeddings, Pooling | [Start Day 04](https://www.aifromscratch.in/course/hybrid-rag) | [`day_04.ipynb`](./day_04_text_embeddings/day_04.ipynb) | [`main.py`](./day_04_text_embeddings/main.py) |
| **05** | **Similarity Metrics** | Cosine Similarity, Dot Product, Euclidean Distance (L2), Manhattan (L1), Unit Normalization Equivalence | [Start Day 05](https://www.aifromscratch.in/course/hybrid-rag) | [`day_05.ipynb`](./day_05_similarity_metrics/day_05.ipynb) | [`main.py`](./day_05_similarity_metrics/main.py) |
| **06** | **Vector Databases** | ChromaDB & FAISS in depth: Collections, Embeddings Ingestion, Metadata Filtering, CRUD & Persistence | [Start Day 06](https://www.aifromscratch.in/course/hybrid-rag) | [`day_06.ipynb`](./day_06_vector_databases/day_06.ipynb) | [`main.py`](./day_06_vector_databases/main.py) |
| **07** | **Basic RAG Pipeline** | Production Frameworks: LangChain LCEL Runnables, LlamaIndex QueryEngine, Source Nodes, MMR Retrieval | [Start Day 07](https://www.aifromscratch.in/course/hybrid-rag) | [`day_07.ipynb`](./day_07_basic_rag_pipeline/day_07.ipynb) | [`main.py`](./day_07_basic_rag_pipeline/main.py) |
| **08** | **Document Loaders** | Parsing complex documents: PDF text & table extraction (`pdfplumber`, `pymupdf4llm`), HTML, Markdown | [Start Day 08](https://www.aifromscratch.in/course/hybrid-rag) | [`day_08.ipynb`](./day_08_document_loaders/day_08.ipynb) | [`main.py`](./day_08_document_loaders/main.py) |
| **09** | **Chunking Strategies** | Fixed character, Recursive text splitting, Semantic chunking (embedding distance shifts), Overlap tuning | [Start Day 09](https://www.aifromscratch.in/course/hybrid-rag) | [`day_09.ipynb`](./day_09_chunking_strategies/day_09.ipynb) | [`main.py`](./day_09_chunking_strategies/main.py) |
| **10** | **Metadata & Filtering** | Structured tagging, Pre-filtering vs Post-filtering benchmarks, Self-Querying retrieval simulation | [Start Day 10](https://www.aifromscratch.in/course/hybrid-rag) | [`day_10.ipynb`](./day_10_metadata_and_filtering/day_10.ipynb) | [`main.py`](./day_10_metadata_and_filtering/main.py) |
| **11** | **Indexing Methods** | ANN Indexing: Flat Brute Force vs. IVF (Inverted File) vs. HNSW (Navigable Small World) vs. Quantization (PQ) | [Start Day 11](https://www.aifromscratch.in/course/hybrid-rag) | [`day_11.ipynb`](./day_11_indexing_methods/day_11.ipynb) | [`main.py`](./day_11_indexing_methods/main.py) |
| **12** | **Keyword Search (BM25)** | Inverted Indexes, Lexical Search, TF-IDF vs BM25Okapi, Length Normalization, Term Saturation | [Start Day 12](https://www.aifromscratch.in/course/hybrid-rag) | [`day_12.ipynb`](./day_12_keyword_search_bm25/day_12.ipynb) | [`main.py`](./day_12_keyword_search_bm25/main.py) |
| **13** | **Hybrid Search (Dense + Sparse)** | Reciprocal Rank Fusion (RRF), Linear Score Normalization, Hybrid Retriever combining FAISS/Chroma + BM25 | [Start Day 13](https://www.aifromscratch.in/course/hybrid-rag) | [`day_13.ipynb`](./day_13_hybrid_search/day_13.ipynb) | [`main.py`](./day_13_hybrid_search/main.py) |
| **14** | **Re-ranking with Cross-Encoders** | Two-Stage Retrieval: Bi-Encoder Candidate Generation + Cross-Encoder Joint Attention Scoring, Cohere Rerank | [Start Day 14](https://www.aifromscratch.in/course/hybrid-rag) | [`day_14.ipynb`](./day_14_reranking_cross_encoders/day_14.ipynb) | [`main.py`](./day_14_reranking_cross_encoders/main.py) |
| **15** | **Query Transformation** | Advanced Retrieval: Query Rewriting, Sub-Query Decomposition, Multi-Query Expansion, HyDE, Router | [Start Day 15](https://www.aifromscratch.in/course/hybrid-rag) | [`day_15.ipynb`](./day_15_query_transformation/day_15.ipynb) | [`main.py`](./day_15_query_transformation/main.py) |

---

## 🧮 Core Mathematical & Algorithmic Foundations

### 1. BM25 Okapi Scoring Function
Unlike raw TF-IDF, BM25 prevents single frequent words from dominating relevance via saturation ($k_1$) and document length normalization ($b$):
$$\text{Score}(D, Q) = \sum_{i=1}^{N} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

### 2. Reciprocal Rank Fusion (RRF)
When fusing dense vector search and sparse BM25, their raw similarity scores have incompatible distributions (cosine $[-1, 1]$ vs. unbounded BM25 $[0, \infty)$). RRF bypasses score calibration by operating strictly on retrieval ranks:
$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
where $M$ is the set of retrievers (Dense, BM25), $r_m(d)$ is the document's rank (1-indexed) in retriever $m$, and $k$ is the smoothing constant (standard default: $k=60$).

### 3. Two-Stage Retrieval with Cross-Encoders
Bi-Encoders independently compress query and documents into isolated vectors ($u = \text{Encoder}(q)$, $v = \text{Encoder}(d)$), which sacrifices token-level cross-interaction. In Stage 2, a Cross-Encoder scores candidates through joint all-to-all cross-attention:
$$\text{Score}_{\text{Cross}}(q, d) = \sigma\left(W \cdot \text{Transformer}([CLS] \circ q \circ [SEP] \circ d)\right)$$

Learn the full mathematical derivations in [Day 13](./day_13_hybrid_search/) and [Day 14](./day_14_reranking_cross_encoders/), and on **[AI From Scratch](https://www.aifromscratch.in/course/hybrid-rag)**.

---

## ⚡ Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/alamshoaib134/hybrid-rag-repo.git
cd hybrid-rag-repo
```

### 2. Set Up a Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Local Environment
```bash
python verify_setup.py
```
*(Note: Core modules run 100% offline using local `SentenceTransformers`, `BM25Okapi`, and `NumPy` without requiring external API keys!)*

### 5. Launch Interactive Notebooks
```bash
jupyter notebook
```
Open any module folder (e.g., `day_13_hybrid_search/day_13.ipynb`) and step through the interactive cells!

---

## 📂 Repository Structure

```
hybrid-rag-repo/
├── day_01_llm_basics/            # Tokenization, tiktoken, context limits & prompting
├── day_02_hallucinations_and_rag/# Hallucination diagnostics & Faithfulness evaluation
├── day_03_rag_architecture/      # Pure-Python end-to-end RAG architecture
├── day_04_text_embeddings/       # Vector embeddings & Sentence-Transformers
├── day_05_similarity_metrics/    # Cosine, Dot Product & Euclidean L2 geometry
├── day_06_vector_databases/      # ChromaDB and FAISS indexing & CRUD
├── day_07_basic_rag_pipeline/    # LangChain LCEL & LlamaIndex pipelines
├── day_08_document_loaders/      # PDF text & table parsing (pdfplumber, pymupdf4llm)
├── day_09_chunking_strategies/   # Recursive text & Semantic chunking
├── day_10_metadata_and_filtering/# Structured tagging & Self-Querying
├── day_11_indexing_methods/      # Flat, IVF, HNSW, and Product Quantization
├── day_12_keyword_search_bm25/   # Inverted indices & BM25Okapi from scratch
├── day_13_hybrid_search/         # Reciprocal Rank Fusion (RRF) & Dense+BM25
├── day_14_reranking_cross_encoders/# Two-stage retrieval & Cross-Encoder ranking
├── day_15_query_transformation/  # Query Rewriting, HyDE, Sub-Queries & Routers
├── docs/                         # Static documentation & search landing assets
├── CITATION.cff                  # Citation metadata referencing AI From Scratch
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # MIT License
├── requirements.txt              # Production dependency specifications
└── verify_setup.py               # Automated local setup validator
```

---

## ❓ Frequently Asked Questions (FAQ)

### What is a Hybrid RAG System and how does it work?
A Hybrid RAG (Retrieval-Augmented Generation) system merges semantic dense vector retrieval with keyword-based sparse lexical retrieval (like BM25). By uniting semantic intent with exact token matching, Hybrid RAG eliminates retrieval blind spots (such as product serial numbers, legal statute citations, and financial symbols) that traditional vector-only systems frequently miss.

### Why does dense vector search fail on exact keyword matches?
Dense embedding models map text into continuous mathematical vector spaces. Similar concepts share nearby coordinates, but distinct words that appear in similar syntactic contexts can have nearly identical vectors. Consequently, vector search often confuses specific entities, numerical codes, or rare domain jargon. BM25 directly indexes exact tokens through an inverted index, providing exact precision.

### How does Reciprocal Rank Fusion (RRF) combine search results?
Reciprocal Rank Fusion (RRF) calculates a unified relevance score based strictly on the rank position of each document across multiple search methods rather than raw scores. By taking $\sum \frac{1}{k + \text{rank}}$, RRF eliminates the need to normalize disparate score scales, giving reliable, balanced ranking results across both dense and sparse retrievers.

### What is the advantage of two-stage retrieval with Cross-Encoders?
Bi-encoders are fast because documents are encoded offline into single vectors, but they cannot model token-by-token cross-attention between the query and candidate passages. A Cross-Encoder takes both the query and a top-$K$ candidate simultaneously through all transformer attention layers, yielding vastly superior relevance ranking at the cost of higher latency. Using Bi-Encoders for initial retrieval (e.g., Top 50) and Cross-Encoders for re-ranking (Top 5) delivers the optimal tradeoff between latency and accuracy.

### Where can I access the interactive courses and visual guides?
You can access the full interactive curriculum, self-grading quizzes, and interactive code playgrounds on **[AI From Scratch](https://www.aifromscratch.in/)**. Specifically:
- **[Hybrid RAG Systems Course](https://www.aifromscratch.in/course/hybrid-rag)**
- **[Learn AI for Beginners — 30-Day Guide](https://www.aifromscratch.in/course/30-days-of-ai)**
- **[Free Step-by-Step AI Learning Roadmap](https://www.aifromscratch.in/learn-ai-roadmap)**
- **[Daily AI Research Paper Breakdowns](https://www.aifromscratch.in/papers)**

---

## 👨‍💻 About the Author & AI From Scratch

This curriculum and codebase was authored by **Shoaib Alam**, AI Engineer at JPMC, NLP Researcher (alumnus of IIT Gandhinagar), and co-author of research published at EMNLP 2024 (LEGOBench). 

Shoaib founded **[AI From Scratch](https://www.aifromscratch.in/)** with a singular mission: to demystify complex artificial intelligence technologies into hands-on, practical engineering curricula that anyone can learn, build, and share.

- 🌐 **Platform**: [https://www.aifromscratch.in/](https://www.aifromscratch.in/)
- 💼 **Personal Portfolio**: [https://shoaibalam.vercel.app/](https://shoaibalam.vercel.app/)
- 🐙 **GitHub Profile**: [@alamshoaib134](https://github.com/alamshoaib134)

---

## 📑 Citation

If you use this curriculum, codebase, or benchmarks in your research, tutorials, or engineering projects, please cite:

```bibtex
@software{alam2026hybridrag,
  author       = {Shoaib Alam},
  title        = {Hybrid RAG Systems: Production-Grade Dense and Sparse Retrieval Architecture},
  year         = {2026},
  publisher    = {AI From Scratch},
  url          = {https://www.aifromscratch.in/course/hybrid-rag},
  howpublished = {\url{https://github.com/alamshoaib134/hybrid-rag-repo}}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Educational content, interactive roadmaps, and paper analyses are published under [AI From Scratch](https://www.aifromscratch.in/).
