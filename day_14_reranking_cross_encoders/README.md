# Day 14: Re-ranking with Cross-Encoders and Cohere Rerank

<div align="center">

[![AI From Scratch](https://img.shields.io/badge/Platform-AI_From_Scratch-6366f1.svg?style=for-the-badge&logo=google-chrome&logoColor=white)](https://www.aifromscratch.in/)
[![Course: Hybrid RAG](https://img.shields.io/badge/Course-Hybrid_RAG_Systems-059669.svg?style=for-the-badge)](https://www.aifromscratch.in/course/hybrid-rag)
[![Free AI Roadmap](https://img.shields.io/badge/Roadmap-Free_AI_Guide-blue.svg?style=for-the-badge)](https://www.aifromscratch.in/learn-ai-roadmap)
[![Daily Papers](https://img.shields.io/badge/Daily_Papers-Research_Simplified-purple.svg?style=for-the-badge)](https://www.aifromscratch.in/papers)

</div>

> 🎓 **Official Companion Lesson for [AI From Scratch](https://www.aifromscratch.in/)**  
> This module is part of the comprehensive **[Hybrid RAG Systems Course](https://www.aifromscratch.in/course/hybrid-rag)**. For interactive quizzes, visual animations, and guided learning pathways, visit **[aifromscratch.in](https://www.aifromscratch.in/)**.

---

## 📌 Module Overview

- **Chapter**: Advanced Retrieval & Fusion  
- **Core Concept**: Sorting retrieved candidates for maximum accuracy with Cross-Encoders  
- **Engineering Focus**: Implements two-stage retrieval. Uses Bi-Encoders for fast initial candidate retrieval (top 50) and Cross-Encoders (sentence-transformers cross-encoder, Cohere Rerank API) for joint token-level cross-attention re-ranking (top 5).

### 🔑 Key Search Topics & Concepts
`cross-encoder reranking` `cohere rerank python` `two-stage retrieval RAG` `bi-encoder vs cross-encoder` `reranker tutorial` `AI From Scratch`

---

## 📂 Contents & Structure

| File | Type | Description |
|:-----|:-----|:------------|
| [`day_14.ipynb`](./day_14.ipynb) | Interactive Notebook | Full theoretical explanation, math derivations, architectural illustrations, and runnable code cells. |
| [`main.py`](./main.py) | Standalone Python Script | Production-ready, runnable implementation of the algorithms and test cases. |

---

## 🚀 How to Run

### 1. Interactive Jupyter Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_14_reranking_cross_encoders/day_14.ipynb
```

### 2. Standalone Python Script
Execute directly from your terminal:
```bash
python day_14_reranking_cross_encoders/main.py
```

---

## 🌐 Continue Learning on AI From Scratch

Master production AI and RAG architecture with structured, interactive courses:
- 🚀 **Full Course**: [Hybrid RAG Systems Interactive Course](https://www.aifromscratch.in/course/hybrid-rag)
- 📘 **Beginner Path**: [Learn AI for Beginners — 30-Day Guide](https://www.aifromscratch.in/course/30-days-of-ai)
- 🗺️ **Visual Pathway**: [Free Step-by-Step AI Learning Roadmap](https://www.aifromscratch.in/learn-ai-roadmap)
- 📑 **Frontier Papers**: [Daily AI Research Papers Explained](https://www.aifromscratch.in/papers)
- 💻 **Main Repository**: [Hybrid RAG Systems on GitHub](https://github.com/alamshoaib134/hybrid-rag-repo)
