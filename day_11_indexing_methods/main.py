"""
Day 11: Indexing Methods (Flat, IVF, HNSW, and Quantization)
Core Concept: How vector databases organize data for fast retrieval
Chapter: Search & Context

This script executes the key algorithmic implementations and examples from Day 11.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 11: Indexing Methods (Flat, IVF, HNSW, and Quantization)")
    print("=" * 60)

# --- Snippet 1 ---
import time
import faiss
import numpy as np

faiss.omp_set_num_threads(1)  # single thread -> fair per-query latency comparisons
rng = np.random.default_rng(0)

N, D, NQ, K = 50_000, 384, 200, 10
TOPICS, SUBTOPICS, SUB_SPREAD, POINT_SPREAD = 100, 1_000, 0.8, 1.6

def f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype="float32")

topic_centers = f32(rng.normal(size=(TOPICS, D)))
faiss.normalize_L2(topic_centers)
sub_centers = f32(topic_centers[rng.integers(0, TOPICS, SUBTOPICS)]
                 + (SUB_SPREAD / np.sqrt(D)) * rng.normal(size=(SUBTOPICS, D)))
faiss.normalize_L2(sub_centers)

def make_points(n: int) -> np.ndarray:
    pts = f32(sub_centers[rng.integers(0, SUBTOPICS, n)] + (POINT_SPREAD / np.sqrt(D)) * rng.normal(size=(n, D)))
    faiss.normalize_L2(pts)
    return pts

xb = make_points(N)       # database vectors ("chunks")
xq = make_points(NQ)      # queries (new points, not copies of the database)

# Ground truth = exact search
flat = faiss.IndexFlatIP(D)
flat.add(xb)
t0 = time.perf_counter()
_, GT = flat.search(xq, K)
flat_ms = (time.perf_counter() - t0) / NQ * 1000

def recall_at_k(found: np.ndarray, gt: np.ndarray = None, k: int = K) -> float:
    gt = GT if gt is None else gt
    return float(np.mean([len(set(found[i, :k]) & set(gt[i, :k])) / k for i in range(len(found))]))

def index_mb(index) -> float:
    return faiss.serialize_index(index).nbytes / 1e6

RESULTS = []

def bench(name: str, index, build_s: float = 0.0) -> dict:
    t0 = time.perf_counter()
    _, found = index.search(xq, K)
    ms = (time.perf_counter() - t0) / NQ * 1000
    row = {"index": name, "recall@10": recall_at_k(found), "ms/query": ms, "MB": index_mb(index), "build_s": build_s}
    RESULTS.append(row)
    print(f"{name:32} recall@10={row['recall@10']:.3f} {ms:7.3f} ms/q {row['MB']:8.1f} MB build {build_s:5.1f}s")
    return row

print(f"Data: {N:,} x {D} | Flat: {flat_ms:.3f} ms/query, {index_mb(flat):.1f} MB, recall 1.000 (by definition)")
RESULTS.append({"index": "Flat (exact)", "recall@10": 1.0, "ms/query": flat_ms, "MB": index_mb(flat), "build_s": 0.0})

# --- Snippet 2 ---
def kmeans(x: np.ndarray, k: int, iters: int = 10) -> np.ndarray:
    cent = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(x @ cent.T, axis=1)    # nearest centroid (cosine on unit vectors)
        for c in range(k):
            members = x[assign == c]
            if len(members):
                cent[c] = members.mean(axis=0)
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    return cent

NLIST = 256
t0 = time.perf_counter()
centroids = kmeans(xb[rng.choice(N, 20_000, replace=False)], NLIST)   # train on a sample
lists = np.argmax(xb @ centroids.T, axis=1)                          # file every vector under a centroid
inverted = {c: np.where(lists == c)[0] for c in range(NLIST)}
print(f"Mini-IVF built in {time.perf_counter() - t0:.1f}s; list sizes: min={min(map(len, inverted.values()))}, "
      f"max={max(map(len, inverted.values()))}")

def mini_ivf_search(q: np.ndarray, nprobe: int, k: int = K) -> np.ndarray:
    probe = np.argsort(-(centroids @ q))[:nprobe]           # nearest neighborhoods
    cand = np.concatenate([inverted[c] for c in probe])
    top = cand[np.argsort(-(xb[cand] @ q))[:k]]             # exact search inside them
    return np.pad(top, (0, k - len(top)), constant_values=-1)

for nprobe in (1, 4, 16, 64):
    t0 = time.perf_counter()
    found = np.stack([mini_ivf_search(q, nprobe) for q in xq])
    ms = (time.perf_counter() - t0) / NQ * 1000
    scanned = nprobe / NLIST
    print(f"mini-IVF nprobe={nprobe:<3} scans ~{scanned:5.1%} of data  recall@10={recall_at_k(found):.3f} {ms:6.2f} ms/q")

# --- Snippet 3 ---
for nlist in (256, 1024):
    ivf = faiss.index_factory(D, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    ivf.train(xb)
    ivf.add(xb)
    build = time.perf_counter() - t0
    for nprobe in (1, 8, 32):
        faiss.ParameterSpace().set_index_parameter(ivf, "nprobe", nprobe)
        bench(f"IVF{nlist},Flat nprobe={nprobe}", ivf, build)

# --- Snippet 4 ---
for M in (16, 32):
    hnsw = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 100
    t0 = time.perf_counter()
    hnsw.add(xb)
    build = time.perf_counter() - t0
    for ef in (16, 64, 256):
        hnsw.hnsw.efSearch = ef
        bench(f"HNSW{M} efSearch={ef}", hnsw, build)

# --- Snippet 5 ---
# (a) SQ8: int8 vectors inside HNSW
sq8 = faiss.index_factory(D, "HNSW32,SQ8", faiss.METRIC_INNER_PRODUCT)
t0 = time.perf_counter()
sq8.train(xb)
sq8.add(xb)
build = time.perf_counter() - t0
faiss.ParameterSpace().set_index_parameter(sq8, "efSearch", 64)
bench("HNSW32,SQ8 efSearch=64", sq8, build)

# (b) PQ + re-ranking with full vectors (RFlat). One build, then sweep the re-rank factor.
pq = faiss.index_factory(D, "IVF256,PQ48,RFlat", faiss.METRIC_INNER_PRODUCT)
t0 = time.perf_counter()
pq.train(xb[:15_000])                  # train codebooks on a sample (faster)
pq.add(xb)
build = time.perf_counter() - t0
faiss.ParameterSpace().set_index_parameter(pq, "nprobe", 32)
for k_factor in (1, 4, 16):           # k_factor=1 $\approx$ PQ ranking alone (no extra candidates)
    faiss.ParameterSpace().set_index_parameter(pq, "k_factor_rf", k_factor)
    bench(f"IVF256,PQ48,RFlat k_factor={k_factor}", pq, build)

pq_only_mb = N * 48 / 1e6
print(f"(PQ codes alone $\approx$ {pq_only_mb:.1f} MB; the RFlat index also stores the full float32 vectors for re-ranking)")

# --- Snippet 6 ---
def to_bits(x: np.ndarray) -> np.ndarray:
    return np.packbits(x > 0, axis=1)          # 384 dims -> 48 bytes

xb_bits, xq_bits = to_bits(xb), to_bits(xq)
print(f"Binary DB size: {xb_bits.nbytes / 1e6:.1f} MB  vs float32 {xb.nbytes / 1e6:.1f} MB")

def binary_search(qi: int, k: int = K, rescore: int = 0) -> np.ndarray:
    ham = np.bitwise_count(np.bitwise_xor(xb_bits, xq_bits[qi])).sum(axis=1)  # Hamming distance
    if rescore:
        cand = np.argpartition(ham, rescore)[:rescore]                         # shortlist by Hamming
        return cand[np.argsort(-(xb[cand] @ xq[qi]))[:k]]                      # re-score with full vectors
    return np.argsort(ham, kind="stable")[:k]

for rescore in (0, 50, 200, 1000):
    t0 = time.perf_counter()
    found = np.stack([binary_search(i, rescore=rescore) for i in range(NQ)])
    ms = (time.perf_counter() - t0) / NQ * 1000
    label = "binary only" if not rescore else f"binary + rescore top {rescore}"
    print(f"{label:<28} recall@10={recall_at_k(found):.3f} {ms:6.2f} ms/q")

# --- Snippet 7 ---
print(f"\n{'index':<42} {'recall@10':>9} {'ms/query':>9} {'MB':>8} {'build s':>8}")
for r in sorted(RESULTS, key=lambda r: r["ms/query"]):
    flag = "  ✅" if r["recall@10"] >= 0.95 else ""
    print(f"{r['index']:<42} {r['recall@10']:>9.3f} {r['ms/query']:>9.3f} {r['MB']:>8.1f} {r['build_s']:>8.1f}{flag}")

TARGET = 0.95
ok = [r for r in RESULTS if r["recall@10"] >= TARGET and r["index"] != "Flat (exact)"]
if ok:
    fastest = min(ok, key=lambda r: r["ms/query"])
    smallest = min(ok, key=lambda r: r["MB"])
    print(f"\nFastest meeting recall $\ge$ {TARGET}: {fastest['index']}")
    print(f"Smallest meeting recall $\ge$ {TARGET}: {smallest['index']}")

# --- Snippet 8 ---
hnsw = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 100
hnsw.add(xb)

print(f"{'selectivity':>11} {'HNSW ef=64':>11} {'HNSW ef=512':>12} {'short(<k)':>10}")
for selectivity in (0.5, 0.05, 0.005):
    allowed = np.where(rng.random(N) < selectivity)[0].astype("int64")  # e.g., one tenant / one product
    # Exact filtered ground truth: brute force on the allowed subset
    sub = xb[allowed]
    gt_sub = allowed[np.argsort(-(xq @ sub.T), axis=1)[:, :K]]

    row = []
    for ef in (64, 512):
        params = faiss.SearchParametersHNSW(sel=faiss.IDSelectorBatch(allowed), efSearch=ef)
        _, found = hnsw.search(xq, K, params=params)
        short = float(np.mean((found == -1).any(axis=1)))
        row.append((recall_at_k(found, gt_sub), short))
    print(f"{selectivity:>11.1%} {row[0][0]:>11.3f} {row[1][0]:>12.3f} {row[0][1]:>10.1%}")

# --- Snippet 9 ---
import chromadb

client = chromadb.PersistentClient(path="./chroma_day11")
col = client.get_or_create_collection(
    "tuned",
    metadata={"hnsw:space": "cosine", "hnsw:M": 32, "hnsw:construction_ef": 200, "hnsw:search_ef": 100},
    embedding_function=None,
)
# Newer Chroma versions also accept a `configuration={"hnsw": {...}}` argument; check your version's docs.

# --- Snippet 10 ---
from qdrant_client import QdrantClient, models

qc = QdrantClient(":memory:")
qc.create_collection(
    "tuned",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True),
    )
)
# At query time: search_params=models.SearchParams(hnsw_ef=128, quantization=models.QuantizationSearchParams(rescore=True))

if __name__ == "__main__":
    print("Starting Day 11 script execution...")
