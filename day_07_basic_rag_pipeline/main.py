"""
Day 07: Build a Basic RAG Pipeline (LangChain and LlamaIndex)
Core Concept: Assembling the pieces with LangChain and LlamaIndex
Chapter: Foundations

This script executes the key algorithmic implementations and examples from Day 7.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 07: Build a Basic RAG Pipeline (LangChain and LlamaIndex)")
    print("=" * 60)

# --- Snippet 1 ---
import os
from pathlib import Path

DATA_DIR = Path("data_day7")
DATA_DIR.mkdir(exist_ok=True)

RAW_DOCS = {
    "loans_2026.txt": ("LOANS", "retail_lending", 2026,
        "ZyntraCorp Bank home loan interest rates start at 7.9% per annum for salaried customers "
        "and 8.3% for self-employed customers. The maximum tenure is 30 years.\n\n"
        "Processing fees are 0.5% of the loan amount, capped at 10,000 rupees."),
    "savings_2026.txt": ("SAVINGS", "retail_banking", 2026,
        "The Savings Plus account offers 4.25% annual interest, credited quarterly. "
        "There is no minimum balance requirement.\n\n"
        "Customers get free unlimited UPI transfers and a free debit card for the first year."),
    "cards_2026.txt": ("CARDS", "cards", 2026,
        "The Platinum credit card has an annual fee of 2,500 rupees, waived on annual spends above 3 lakh rupees.\n\n"
        "It offers 2% cashback on online purchases and airport lounge access four times per year."),
    "fd_2026.txt": ("FD", "retail_banking", 2026,
        "Fixed deposit rates are 7.1% for tenures of 1 to 2 years and 7.4% for senior citizens.\n\n"
        "Premature withdrawal attracts a penalty of 1% on the applicable rate."),
    "support_2026.txt": ("SUPPORT", "operations", 2026,
        "Customer support is available 24/7 via phone at 1800-000-000 and through in-app chat.\n\n"
        "Lost card reports are handled immediately through the mobile app."),
    "loans_2023.txt": ("LOANS-OLD", "retail_lending", 2023,
        "ZyntraCorp Bank home loan interest rates start at 8.6% per annum for all customers. "
        "This policy applied during 2023.")
}

for fname, (_, _, _, text) in RAW_DOCS.items():
    (DATA_DIR / fname).write_text(text, encoding="utf-8")

# Metadata we'll attach after loading (in real life: from a document registry or file properties)
FILE_META = {fname: {"doc_id": d, "department": dept, "year": yr} for fname, (d, dept, yr, _) in RAW_DOCS.items()}

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Same evaluation set as Day 3
EVAL_SET = [
    ("home loan rate for salaried people", "LOANS"),
    ("processing fee on home loan", "LOANS"),
    ("savings account interest rate", "SAVINGS"),
    ("credit card annual fee waiver", "CARDS"),
    ("fixed deposit rate for senior citizens", "FD"),
    ("how do I report a lost card", "SUPPORT"),
    ("cost of borrowing to buy a house", "LOANS"),
    ("do I have to keep money parked in my account", "SAVINGS"),
    ("helpline number", "SUPPORT"),
    ("breaking my term deposit early", "FD")
]

print(f"Wrote {len(RAW_DOCS)} files to {DATA_DIR.resolve()}")

# --- Snippet 2 ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1) LOAD (plain Python + business metadata; Day 8 covers PDF/HTML/DOCX parsing)
docs = [
    Document(page_content=p.read_text(encoding="utf-8"),
             metadata={"source": str(p), **FILE_META[p.name]})
    for p in sorted(DATA_DIR.glob("*.txt"))
]
print(f"Loaded {len(docs)} documents. Example metadata: {docs[0].metadata}")

# --- Snippet 3 ---
# 2) SPLIT (chunk_size is in CHARACTERS here)
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30, add_start_index=True)
chunks = splitter.split_documents(docs)

# Deterministic chunk IDs: doc_id#c{n}
counters = {}
for c in chunks:
    n = counters.get(c.metadata["doc_id"], 0)
    counters[c.metadata["doc_id"]] = n + 1
    c.metadata["chunk_id"] = f"{c.metadata['doc_id']}#c{n}"
print(f"Split into {len(chunks)} chunks:")
for c in chunks[:4]:
    print(f"  {c.metadata['chunk_id']:<14} {len(c.page_content):>3} chars | {c.page_content[:60]}...")

# 3) EMBED + STORE
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
vectorstore = Chroma(
    collection_name="bank_lc",
    embedding_function=embeddings,
    persist_directory="./chroma_day7",
    collection_metadata={"hnsw:space": "cosine"}
)
vectorstore.reset_collection()                         # lab only: start fresh each run
vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])

# 4) RETRIEVE (inspect before generating!)
for doc, score in vectorstore.similarity_search_with_relevance_scores("cost of borrowing to buy a house", k=3):
    print(f"  [{score:.3f}] {doc.metadata['chunk_id']:<14} {doc.page_content[:60]}...")

# --- Snippet 4 ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

SYSTEM = """You are ZyntraCorp Bank's assistant.
Answer ONLY using the CONTEXT. If the answer is not in the CONTEXT, reply exactly:
"I don't know based on the provided documents."
Cite chunk IDs in square brackets, e.g., [LOANS#c0]. Be concise."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "CONTEXT:\n\"\"\"\n{context}\n\"\"\"\n\nQUESTION: {question}"),
])

def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata['chunk_id']}] {d.page_content}" for d in docs)

# LLM: real if an API key is set, otherwise a transparent echo "LLM"
if USE_OPENAI:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    llm = RunnableLambda(lambda prompt_value: "[MOCK LLM] Would answer from:\n"
                         + prompt_value.to_messages()[-1].content[:400])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"year": 2026}})

# (a) Simple chain: question -> answer
rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke("What is the home loan rate for salaried customers?"))

# (b) Chain that ALSO returns the sources (needed for citations and audit)
answer_step = RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]}) \
    | prompt | llm | StrOutputParser()
rag_with_sources = RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=answer_step)

out = rag_with_sources.invoke("Is there a minimum balance on the savings account?")
print("\nANSWER :", out["answer"][:300])
print("SOURCES:", [d.metadata["chunk_id"] for d in out["docs"]])

# (c) Streaming: tokens appear as they're generated (real LLMs stream token by token)
print("\nSTREAM :", end=" ")
for piece in rag_chain.stream("What is the FD rate for senior citizens?"):
    print(piece, end="", flush=True)
print()

# (d) Batch: several questions at once
answers = rag_chain.batch(["What is the Platinum card fee?", "What is ZyntraCorp's share price?"])
for a in answers:
    print("\nBATCH  :", a[:200])

# --- Snippet 5 ---
q = "Tell me about home loans"
variants = {
    "similarity k=3": vectorstore.as_retriever(search_kwargs={"k": 3}),
    "similarity + year=2026": vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"year": 2026}}),
    "mmr (diverse)": vectorstore.as_retriever(search_type="mmr",
                                              search_kwargs={"k": 3, "fetch_k": 8,
                                                             "lambda_mult": 0.5}),
    "score threshold 0.3": vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 5}),
    "AND filter": vectorstore.as_retriever(search_kwargs={
        "k": 3, "filter": {"$and": [{"year": 2026}, {"department": "retail_lending"}]}}),
}
for name, r in variants.items():
    print(f"{name:<24} -> {[d.metadata['chunk_id'] for d in r.invoke(q)]}")

# --- Snippet 6 ---
from llama_index.core import (Document, PromptTemplate, Settings, StorageContext,
                              VectorStoreIndex, load_index_from_storage)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1) Global settings: be EXPLICIT (otherwise LlamaIndex defaults to OpenAI for both!)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.node_parser = SentenceSplitter(chunk_size=80, chunk_overlap=10)   # TOKENS, not characters
if USE_OPENAI:
    from llama_index.llms.openai import OpenAI
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
else:
    from llama_index.core.llms import MockLLM
    Settings.llm = MockLLM(max_tokens=40)                                  # placeholder text generator

# 2) Documents with metadata
li_docs = [
    Document(text=text, metadata={"doc_id": d, "department": dept, "year": yr, "file": fname},
             id_=d)
    for fname, (d, dept, yr, text) in RAW_DOCS.items()
]

# 3) Index (parse -> nodes -> embed -> store)
index = VectorStoreIndex.from_documents(li_docs)
print("Nodes indexed:", len(index.docstore.docs))

# 4) Our own grounded prompt (never rely on the default for RAG)
qa_prompt = PromptTemplate(
    "You are ZyntraCorp Bank's assistant. Answer ONLY using the context below.\n"
    "If the answer is not in the context, reply exactly: \"I don\'t know based on the provided documents.\"\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Question: {query_str}\nAnswer: "
)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    filters=MetadataFilters(filters=[ExactMatchFilter(key="year", value=2026)]),
    text_qa_template=qa_prompt,
)
response = query_engine.query("What is the home loan rate for salaried customers?")
print("ANSWER:", str(response)[:200])
for sn in response.source_nodes:
    print(f"  [{sn.score:.3f} {sn.node.metadata['doc_id']:<10} {sn.node.get_content()[:60]}...]")

# 5) Retriever only (no LLM), which is useful for evaluation
li_retriever = index.as_retriever(similarity_top_k=3)
print([n.node.metadata["doc_id"] for n in li_retriever.retrieve("cost of borrowing to buy a house")])

# 6) Persist and reload (no re-embedding needed next time)
index.storage_context.persist(persist_dir="./storage_day7")
reloaded = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage_day7"))
print("Reloaded nodes:", len(reloaded.docstore.docs))

# --- Snippet 7 ---
def recall_mrr(retrieve_doc_ids, eval_set, k=3):
    """retrieve_doc_ids(question) -> ordered list of doc_ids (length <= k)."""
    hits, rr = 0, 0.0
    for question, relevant in eval_set:
        ids = retrieve_doc_ids(question)[:k]
        if relevant in ids:
            hits += 1
            rr += 1 / (ids.index(relevant) + 1)
    n = len(eval_set)
    return {"recall@k": round(hits / n, 2), "mrr": round(rr / n, 2)}

lc_ret = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"year": 2026}})
li_ret = index.as_retriever(similarity_top_k=3,
                            filters=MetadataFilters(filters=[ExactMatchFilter(key="year",
                            value=2026)]))

print("LangChain :", recall_mrr(lambda q: [d.metadata["doc_id"] for d in lc_ret.invoke(q)], EVAL_SET))
print("LlamaIndex:", recall_mrr(lambda q: [n.node.metadata["doc_id"] for n in li_ret.retrieve(q)], EVAL_SET))

# From scratch (Day 3 + Day 4 classes), if they're loaded in this session:
# rag = RAGPipeline(WordChunker(25, 5), SentenceTransformerEmbedder(), InMemoryVectorStore(),
#                   EchoGenerator(), k=3, min_score=0.0)
# rag.index(DOCS)
# print("Scratch   :", recall_mrr(lambda q: [c.doc_id for c, _ in rag.retriever.retrieve(q, k=3, filters={"year": 2026})], EVAL_SET))

# --- Snippet 8 ---
import argparse

REFUSAL = "I don't know based on the provided documents."
MIN_SCORE = 0.30  # tune on answerable vs unanswerable questions (Day 5)

def answer(question: str) -> dict:
    hits = vectorstore.similarity_search_with_relevance_scores(question, k=3, filter={"year": 2026})
    if not hits or hits[0][1] < MIN_SCORE:
        return {"answer": REFUSAL, "sources": [], "top_score": hits[0][1] if hits else None}
    docs = [d for d, _ in hits]
    text = (prompt | llm | StrOutputParser()).invoke({"context": format_docs(docs), "question": question})
    return {"answer": text, "sources": [d.metadata["chunk_id"] for d in docs], "top_score": round(hits[0][1], 3)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.eval:
        print(recall_mrr(lambda q: [d.metadata["doc_id"] for d in lc_ret.invoke(q)], EVAL_SET))
    else:
        while (q := input("\nAsk (or 'exit'): ").strip().lower()) != "exit":
            r = answer(q)
            print(f"\n{r['answer']}\nSources: {r['sources']} | top score: {r['top_score']}")

if __name__ == "__main__":
    print("Starting Day 07 script execution...")
