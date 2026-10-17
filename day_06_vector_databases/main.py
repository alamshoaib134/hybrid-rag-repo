"""
Day 06: Vector Databases (FAISS, Chroma, Pinecone, pgvector)
Core Concept: The professional warehouse for embeddings: storage, filtering, and retrieval
Chapter: Foundations

This script executes the key algorithmic implementations and examples from Day 6.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 06: Vector Databases (FAISS, Chroma, Pinecone, pgvector)")
    print("=" * 60)

# --- Snippet 1 ---
import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalized float32 embeddings (cosine = dot product)."""
    return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype="float32")

DIM = embed(["dimension probe"]).shape[1]   # 384 for MiniLM

CHUNKS = [
    {"id": "LOANS#c0", "doc_id": "LOANS", "department": "retail_lending", "year": 2026, 
     "text": "Home loan interest rates start at 7.9% per annum for salaried customers and 8.3% for self-employed customers."},
    {"id": "LOANS#c1", "doc_id": "LOANS", "department": "retail_lending", "year": 2026, 
     "text": "Home loan processing fees are 0.5% of the loan amount, capped at 10,000 rupees. Maximum tenure is 30 years."},
    {"id": "SAVINGS#c0", "doc_id": "SAVINGS", "department": "retail_banking", "year": 2026, 
     "text": "The Savings Plus account offers 4.25% annual interest, credited quarterly, with no minimum balance requirement."},
    {"id": "CARDS#c0", "doc_id": "CARDS", "department": "cards", "year": 2026, 
     "text": "The Platinum credit card has an annual fee of 2,500 rupees, waived on annual spends above 3 lakh rupees."},
    {"id": "CARDS#c1", "doc_id": "CARDS", "department": "cards", "year": 2026, 
     "text": "Platinum cardholders get 2% cashback on online purchases and four airport lounge visits per year."},
    {"id": "FD#c0", "doc_id": "FD", "department": "retail_banking", "year": 2026, 
     "text": "Fixed deposit rates are 7.1% for 1 to 2 year tenures and 7.4% for senior citizens."},
    {"id": "FD#c1", "doc_id": "FD", "department": "retail_banking", "year": 2026, 
     "text": "Premature withdrawal of a fixed deposit attracts a penalty of 1% on the applicable rate."},
    {"id": "SUPPORT#c0", "doc_id": "SUPPORT", "department": "operations", "year": 2026, 
     "text": "Customer support is available 24/7 by phone at 1800-000-000 and through in-app chat."},
    {"id": "LOANS-OLD#c0", "doc_id": "LOANS-OLD", "department": "retail_lending", "year": 2023, 
     "text": "In 2023, home loan interest rates started at 8.6% per annum for all customers."},
]

texts = [c["text"] for c in CHUNKS]
vecs = embed(texts)
print("Embeddings:", vecs.shape, vecs.dtype)

# --- Snippet 2 ---
import faiss

# 1) Build an exact inner-product index with our own int64 IDs
index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
index.add_with_ids(vecs, np.arange(len(CHUNKS), dtype="int64"))
print("Vectors in FAISS:", index.ntotal)

def faiss_search(query: str, k: int = 3):
    scores, ids = index.search(embed([query]), k)
    # FAISS returns only integer IDs, so map them back to our chunks ourselves
    return [(CHUNKS[i]["id"], round(float(s), 3)) for s, i in zip(scores[0], ids[0]) if i != -1]

print(faiss_search("cost of borrowing to buy a house"))
print(faiss_search("what is the penalty for breaking an FD early?"))

# 2) Persist and reload (vectors only! keep CHUNKS in your own store, e.g., JSON/SQLite)
faiss.write_index(index, "bank.faiss")
reloaded = faiss.read_index("bank.faiss")
print("Reloaded vectors:", reloaded.ntotal)

**Post-filtering and its pitfall:**

def faiss_search_filtered(query: str, k: int = 3, year: int | None = None, overfetch: int = 1):
    fetch = min(index.ntotal, k * overfetch)
    scores, ids = index.search(embed([query]), fetch)
    results = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        if year is not None and CHUNKS[i]["year"] != year:
            continue                             # post-filter
        results.append((CHUNKS[i]["id"], round(float(s), 3)))
        if len(results) == k:
            break
    return results

q = "credit card annual fee and cashback"
print("year=2023, overfetch=1 :", faiss_search_filtered(q, k=2, year=2023, overfetch=1))
print("year=2023, overfetch=10:", faiss_search_filtered(q, k=2, year=2023, overfetch=10))

# --- Snippet 3 ---
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Start fresh each run (fine for a lab; never do this casually in production!)
if "bank_docs" in [c.name for c in client.list_collections()]:
    client.delete_collection("bank_docs")

col = client.create_collection(
    name="bank_docs",
    metadata={"hnsw:space": "cosine"},      # default is L2! (see Day 5)
    embedding_function=None,                # we supply our own embeddings
)

col.add(
    ids=[c["id"] for c in CHUNKS],
    embeddings=vecs.tolist(),
    documents=texts,
    metadatas=[{"doc_id": c["doc_id"], "department": c["department"], "year": c["year"]} for c in CHUNKS],
)
print("Records in Chroma:", col.count())

def chroma_search(query: str, k: int = 3, where: dict | None = None, where_document: dict | None = None):
    res = col.query(
        query_embeddings=embed([query]).tolist(),
        n_results=k,
        where=where,
        where_document=where_document,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma returns DISTANCES; for cosine space: similarity = 1 - distance
    return [(i, round(1 - d, 3)) for i, d in zip(res["ids"][0], res["distances"][0])]

print("No filter       :", chroma_search("home loan interest rate"))
print("year = 2026     :", chroma_search("home loan interest rate", where={"year": 2026}))
print("year = 2023     :", chroma_search("credit card annual fee", where={"year": 2023}))  # pre-filter: still finds it
print("AND filter      :", chroma_search(
    "interest rate", 
    where={"$and": [{"year": {"$gte": 2025}}, {"department": {"$in": ["retail_banking", "retail_lending"]}}]},
))
print("Doc text contains:", chroma_search("rates", where_document={"$contains": "senior"}))

# --- Snippet 4 ---
import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def reindex_document(collection, doc_id: str, new_chunk_texts: list[str], metadata: dict):
    """Delete all old chunks of a document, then add the new ones with deterministic IDs."""
    collection.delete(where={"doc_id": doc_id})
    ids = [f"{doc_id}#c{n}" for n in range(len(new_chunk_texts))]
    collection.add(
        ids=ids,
        embeddings=embed(new_chunk_texts).tolist(),
        documents=new_chunk_texts,
        metadatas=[{**metadata, "doc_id": doc_id, "hash": content_hash(t)} for t in new_chunk_texts],
    )
    return ids

# The loan policy was revised: the rate dropped and the doc now has ONE chunk instead of two
new_ids = reindex_document(
    col, "LOANS", 
    ["Revised policy: home loan interest rates now start at 7.6% for all customers."], 
    {"department": "retail_lending", "year": 2026},
)
print("New IDs:", new_ids)
print("LOANS chunks now:", col.get(where={"doc_id": "LOANS"})["ids"])      # the old LOANS#c1 is gone
print("Search:", chroma_search("home loan interest rate", where={"year": 2026}))

# Retire the outdated 2023 policy entirely
col.delete(where={"doc_id": "LOANS-OLD"})
print("Total records:", col.count())

# Persistence check: a brand-new client sees the same data
client2 = chromadb.PersistentClient(path="./chroma_db")
print("Records after reopening:", client2.get_collection("bank_docs").count())

# --- Snippet 5 ---
class FaissVectorStore:
    """Exact inner-product search with over-fetch + post-filtering for metadata."""
    
    def __init__(self, overfetch: int = 5):
        self.index = None
        self.chunks = []
        self.overfetch = overfetch
        
    def add(self, chunks, vectors):
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(chunks)
        
    def search(self, query_vec, k=3, filters=None):
        q = np.ascontiguousarray(np.asarray(query_vec, dtype="float32").reshape(1, -1))
        fetch = min(len(self.chunks), k * self.overfetch if filters else k)
        scores, ids = self.index.search(q, fetch)
        results = []
        for s, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            chunk = self.chunks[i]
            if filters and any(chunk.metadata.get(key) != val for key, val in filters.items()):
                continue
            results.append((chunk, float(s)))
            if len(results) == k:
                break
        return results

class ChromaVectorStore:
    """Persistent Chroma collection using cosine space; metadata filters are pushed down to Chroma."""
    
    def __init__(self, path: str = "./chroma_db", name: str = "day6_pipeline"):
        self.client = chromadb.PersistentClient(path=path)
        if name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name)
        self.col = self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"}, 
                                                 embedding_function=None)
        self._chunks = {}
        
    def add(self, chunks, vectors):
        self.col.add(
            ids=[c.id for c in chunks],
            embeddings=np.asarray(vectors, dtype="float32").tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{**c.metadata, "doc_id": c.doc_id} for c in chunks],
        )
        self._chunks.update({c.id: c for c in chunks})
        
    def search(self, query_vec, k=3, filters=None):
        where = None
        if filters:
            conds = [{key: val} for key, val in filters.items()]
            where = conds[0] if len(conds) == 1 else {"$and": conds}
        res = self.col.query(query_embeddings=[np.asarray(query_vec, dtype="float32").tolist()], 
                             n_results=k, where=where)
        return [(self._chunks[i], 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

# --- Snippet 6 ---
# Re-use from Day 3: Document, DOCS, WordChunker, InMemoryVectorStore, EchoGenerator,
#                    RAGPipeline, EVAL_SET, evaluate_retrieval
# Re-use from Day 4: SentenceTransformerEmbedder

for name, store in [
    ("InMemory", InMemoryVectorStore()),
    ("FAISS", FaissVectorStore()),
    ("Chroma", ChromaVectorStore()),
]:
    rag = RAGPipeline(WordChunker(25, 5), SentenceTransformerEmbedder(), store,
                      EchoGenerator(), k=3, min_score=0.2)
    rag.index(DOCS)
    print(f"{name:<9}", evaluate_retrieval(rag.retriever, EVAL_SET, k=3, filters={"year": 2026}))

# --- Snippet 7 ---
import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect("postgresql://postgres:postgres@localhost:5432/postgres",
                       autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

conn.execute("DROP TABLE IF EXISTS chunks")
conn.execute(f"""
    CREATE TABLE chunks (
        id          text PRIMARY KEY,
        doc_id      text NOT NULL,
        department  text,
        year        int,
        content     text NOT NULL,
        embedding   vector({DIM}) NOT NULL
    )
""")

with conn.cursor() as cur:
    for c, v in zip(CHUNKS, vecs):
        cur.execute(
            "INSERT INTO chunks (id, doc_id, department, year, content, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
            (c["id"], c["doc_id"], c["department"], c["year"], c["text"], v),
        )

conn.execute("CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)")
conn.execute("CREATE INDEX ON chunks (year)")

def pg_search(query: str, k: int = 3, year: int | None = None):
    q = embed([query])[0]
    sql = """
        SELECT id, content, 1 - (embedding <=> %(q)s) AS cosine_similarity
        FROM chunks
        WHERE %(year)s::int IS NULL OR year = %(year)s::int
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
    """
    return conn.execute(sql, {"q": q, "year": year, "k": k}).fetchall()

for row in pg_search("cost of borrowing to buy a house", year=2026):
    print(row[0], round(row[2], 3), row[1][:60])

# --- Snippet 8 ---
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = "bank-docs-day6"

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME, dimension=DIM, metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pidx = pc.Index(INDEX_NAME)
pidx.upsert(
    vectors=[
        {"id": c["id"], "values": v.tolist(),
         "metadata": {"doc_id": c["doc_id"], "department": c["department"], "year": c["year"], "text": c["text"]}}
        for c, v in zip(CHUNKS, vecs)
    ],
    namespace="zyntracorp",
)

res = pidx.query(
    vector=embed(["cost of borrowing to buy a house"])[0].tolist(),
    top_k=3,
    include_metadata=True,
    filter={"year": {"$eq": 2026}},
    namespace="zyntracorp",
)

for m in res.matches:
    print(m.id, round(m.score, 3), m.metadata["text"][:60])

# Clean up to stay within free-tier limits:
# pc.delete_index(INDEX_NAME)

# --- Snippet 9 ---
import time

N, D, NQ, K = 50_000, 384, 200, 10
rng = np.random.default_rng(0)

def make_data(structured: bool):
    """Structured = clustered like real embeddings (topics); random = no structure (worst case)."""
    if structured:
        centers = rng.normal(size=(500, D)).astype("float32")
        faiss.normalize_L2(centers)
        pts = centers[rng.integers(0, 500, N + NQ)] + (0.3 / np.sqrt(D)) * rng.normal(size=(N + NQ, D)).astype("float32")
    else:
        pts = rng.normal(size=(N + NQ, D)).astype("float32")
    pts = np.ascontiguousarray(pts, dtype="float32")
    faiss.normalize_L2(pts)
    return pts[:N], np.ascontiguousarray(pts[N:])

for label, structured in [("STRUCTURED (clustered)", True), ("RANDOM (no structure)", False)]:
    data, queries = make_data(structured)
    
    flat = faiss.IndexFlatIP(D)                        # exact search = ground truth
    flat.add(data)
    t0 = time.perf_counter()
    _, gt = flat.search(queries, K)
    t_flat = time.perf_counter() - t0
    
    hnsw = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    hnsw.add(data)
    t_build = time.perf_counter() - t0
    
    print(f"\n=== {label} ===")
    print(f"Exact (flat) : {t_flat / NQ * 1000:.3f} ms/query, recall@{K} = 1.000")
    print(f"HNSW build   : {t_build:.1f} s")
    
    for ef in (16, 64, 256):
        hnsw.hnsw.efSearch = ef
        t0 = time.perf_counter()
        _, approx = hnsw.search(queries, K)
        t_h = time.perf_counter() - t0
        
        recall = np.mean([len(set(gt[i]) & set(approx[i])) / K for i in range(NQ)])
        print(f"HNSW ef={ef:<4}: {t_h / NQ * 1000:.3f} ms/query, recall@{K} = {recall:.3f}")

if __name__ == "__main__":
    print("Starting Day 06 script execution...")
