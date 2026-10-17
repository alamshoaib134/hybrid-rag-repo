"""
Environment Verification Script for Hybrid RAG Systems
Tests whether essential dependencies and environments are configured properly.
"""
from __future__ import annotations
import sys

PACKAGES = [
    ("numpy", "NumPy (Math & Vector Operations)"),
    ("tiktoken", "Tiktoken (Token Counting)"),
    ("sentence_transformers", "SentenceTransformers (Dense Embeddings)"),
    ("chromadb", "ChromaDB (Vector Database)"),
    ("faiss", "FAISS (Vector Indexing)"),
    ("rank_bm25", "Rank-BM25 (Sparse Keyword Search)"),
    ("langchain", "LangChain (LCEL Pipelines)"),
    ("llama_index", "LlamaIndex (Data Framework)"),
    ("pypdf", "PyPDF (Document Loading)"),
    ("openai", "OpenAI SDK (LLM Generation)"),
]

def check_env():
    print("=" * 65)
    print(" Hybrid RAG Systems — Environment Verification")
    print("=" * 65)
    print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")
    print("-" * 65)

    passed = 0
    missing = []
    for pkg, name in PACKAGES:
        try:
            __import__(pkg)
            print(f" [PASS] {name}")
            passed += 1
        except ImportError:
            print(f" [FAIL] {name} (Missing '{pkg}')")
            missing.append(pkg)

    print("-" * 65)
    print(f"Result: {passed}/{len(PACKAGES)} components ready.")
    if missing:
        print(f"\nTo install missing dependencies, run:")
        print(f"  pip install -r requirements.txt")
    else:
        print("\nAll core dependencies are installed! You are ready to start.")
    print("=" * 65)

if __name__ == "__main__":
    check_env()
