"""
Day 12: Keyword Search (BM25 and Sparse Retrieval)
Core Concept: Classic text search vs. Semantic search
Chapter: Search & Context

This script executes the key algorithmic implementations and examples from Day 12.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 12: Keyword Search (BM25 and Sparse Retrieval)")
    print("=" * 60)

# --- Snippet 1 ---
import math
import re
from collections import Counter, defaultdict

import numpy as np

DOCS = [
    ("HL-RATES", "Home Loan Interest Rates",
     "Floating home loan interest rates start at 7.9% per annum for salaried customers and 8.3% for self-employed customers."),
    ("HL-FLEX", "Flexi Home Loan (ZYN-HL-FLEX)",
     "The ZYN-HL-FLEX flexi home loan links an overdraft account to your loan. Surplus funds parked in it reduce interest, and you can withdraw them anytime."),
    ("HL-TAX", "Home Loan Tax Benefits",
     "Principal repayment on a home loan qualifies for deduction under Section 80C up to 1.5 lakh rupees, and interest paid qualifies under Section 24(b) up to 2 lakh rupees for a self-occupied property."),
    ("HL-FEES", "Home Loan Fees",
     "A processing fee of 0.5% of the loan amount is charged, capped at 10,000 rupees. There are no prepayment charges on floating rate loans."),
    ("FD-RATES", "Fixed Deposit Rates",
     "Fixed deposit rates are 7.1% for 1 to 2 years. Senior citizens earn 7.4% on the same tenure."),
    ("FD-TDS", "TDS on Fixed Deposits",
     "TDS is deducted on FD interest above 40,000 rupees a year (50,000 for senior citizens). Submit Form 15G or Form 15H to avoid TDS if your income is below the taxable limit. TDS details appear in Form 26AS."),
    ("SAV-MAB", "Savings Plus Account",
     "The Savings Plus account has no minimum balance requirement and pays 4.25% interest credited quarterly."),
    ("UPI-ERR", "UPI Error Codes",
     "Error E1043 means the UPI transaction failed due to a bank server timeout; retry after 5 minutes. Error E2091 means the daily UPI limit of 1 lakh rupees has been exceeded."),
    ("BR-PUNE", "Pune Main Branch",
     "The Pune Main Branch is at FC Road, Shivajinagar. IFSC code: ZYNT0004521. MICR code: 411240017."),
    ("BR-MUMBAI", "Mumbai Fort Branch",
     "The Mumbai Fort Branch is at Horniman Circle. IFSC code: ZYNT0001187."),
    ("SWIFT", "International Transfers",
     "For inward international wire transfers, use SWIFT code ZYNTINBBXXX along with the beneficiary's account number."),
    ("NEFT", "NEFT and RTGS",
     "NEFT transfers are available 24x7 including holidays. RTGS is used for amounts of 2 lakh rupees and above."),
    ("CARD-LOST", "Lost or Stolen Cards",
     "If your card is lost or stolen, open the mobile app, go to Cards, and tap Block Card. The card is blocked instantly."),
    ("CARD-PLAT", "Platinum Credit Card",
     "The Platinum credit card has an annual fee of 2,500 rupees, waived on spends above 3 lakh rupees a year. There is no minimum spend requirement, and an outstanding balance can be converted into EMIs."),
    ("GRIEV", "Grievance Redressal",
     "If a complaint is unresolved after 30 days, write to the Nodal Officer, Ms. Radhika Iyer, at nodal.officer@zyntracorp.example, or escalate to the RBI Banking Ombudsman."),
    ("SUPPORT", "Customer Support",
     "Customer care is available 24/7 at 1800-000-000 and via in-app chat."),
]

IDS = [d[0] for d in DOCS]
FULL_TEXT = [f"{title}. {text}" for _, title, text in DOCS]

# (question, relevant doc, query type)
QUERIES = [
    ("What is the IFSC code of the Pune branch?", "BR-PUNE", "exact"),
    ("Which branch has IFSC ZYNT0001187?", "BR-MUMBAI", "exact"),
    ("What does UPI error E1043 mean?", "UPI-ERR", "exact"),
    ("What is ZYN-HL-FLEX?", "HL-FLEX", "exact"),
    ("Section 80C deduction limit", "HL-TAX", "exact"),
    ("When should I submit Form 15H?", "FD-TDS", "exact"),
    ("SWIFT code for receiving money from abroad", "SWIFT", "exact"),
    ("Who is Radhika Iyer?", "GRIEV", "exact"),
    ("How much does it cost to borrow for buying a house?", "HL-RATES", "paraphrase"),
    ("I misplaced my plastic, how do I freeze it?", "CARD-LOST", "paraphrase"),
    ("Whom do I phone for help?", "SUPPORT", "paraphrase"),
    ("What do elderly people earn on term deposits?", "FD-RATES", "paraphrase"),
    ("Where can I complain if the bank ignores me?", "GRIEV", "paraphrase"),
    ("Do I need to maintain some amount of money to avoid fees?", "SAV-MAB", "paraphrase"),
]

print(len(DOCS), "docs,", len(QUERIES), "queries")

# --- Snippet 2 ---
try:
    import snowballstemmer
    _stemmer = snowballstemmer.stemmer("english")
    stem = _stemmer.stemWord
except ImportError:
    # graceful fallback
    stem = lambda w: re.sub(r"(ing|ed|es|s)$", "", w) if len(w) > 4 else w

STOPWORDS = {"a", "an", "the", "is", "are", "of", "to", "in", "on", "for", "and", "or", "at", "by", "it",
             "what", "which", "who", "whom", "how", "do", "does", "i", "my", "me", "can", "if", "your",
             "with", "be", "this", "that", "from", "has", "have", "will", "should", "when", "where", "there"}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./@][a-z0-9]+)*%?")

def naive_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def analyze(text: str) -> list[str]:
    terms = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        is_code = any(ch.isdigit() for ch in tok) or "-" in tok or "@" in tok or "." in tok
        if is_code:
            terms.append(tok)                            # keep the code whole: zyn-hl-flex, e1043, 7.9%
            parts = [p for p in re.split(r"[-./@]", tok) if p and p != tok]
            terms.extend(parts)                          # also emit parts: zyn, hl, flex
        else:
            terms.append(stem(tok))                      # stem ordinary words only
    return terms

for sample in ["What is ZYN-HL-FLEX?", "Interest rates start at 7.9% per annum",
               "Submit Form 15G/15H", "Charges charged charging", "email nodal.officer@zyntracorp.example"]:
    print(f"{sample!r:45}\n  naive   : {naive_tokens(sample)}\n  analyzed: {analyze(sample)}")

# --- Snippet 3 ---
class BM25:
    def __init__(self, docs_terms: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_terms)
        self.doc_len = [len(t) for t in docs_terms]
        self.avgdl = sum(self.doc_len) / self.N
        self.tf = [Counter(t) for t in docs_terms]
        self.postings = defaultdict(list)               # term -> [doc indices] (inverted index)
        for i, counts in enumerate(self.tf):
            for term in counts:
                self.postings[term].append(i)
        self.idf = {t: math.log((self.N - len(p) + 0.5) / (len(p) + 0.5) + 1) for t, p in self.postings.items()}

    def term_score(self, term: str, i: int) -> float:
        f = self.tf[i].get(term, 0)
        if f == 0:
            return 0.0
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
        return self.idf[term] * f * (self.k1 + 1) / (f + norm)

    def search(self, query_terms: list[str], k: int = 3) -> list[tuple[int, float]]:
        scores = defaultdict(float)
        for term in set(query_terms):
            for i in self.postings.get(term, []):        # only docs that contain the term
                scores[i] += self.term_score(term, i)
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    def explain(self, query_terms: list[str], i: int) -> dict:
        return {t: round(self.term_score(t, i), 3) for t in set(query_terms) if self.term_score(t, i) > 0}


bm25 = BM25([analyze(t) for t in FULL_TEXT])
print(f"Vocabulary: {len(bm25.postings)} terms | avg doc length: {bm25.avgdl:.1f} terms")

for q in ["What is the IFSC code of the Pune branch?", "What does UPI error E1043 mean?", "Who is Radhika Iyer?"]:
    qt = analyze(q)
    print(f"\nQ: {q}\n  terms: {qt}")
    for i, s in bm25.search(qt):
        print(f"  {s:6.2f} {IDS[i]:<10} why: {bm25.explain(qt, i)}")

print("\nIDF of a few terms (rare = high):")
for t in ["zynt0004521", "ifsc", "pune", "loan", "rupe"]:
    if t in bm25.idf:
        print(f"  {t:<12} df={len(bm25.postings[t]):>2}  idf={bm25.idf[t]:.2f}")

# --- Snippet 4 ---
def tf_part(f: int, k1: float, len_ratio: float = 1.0, b: float = 0.75) -> float:
    return f * (k1 + 1) / (f + k1 * (1 - b + b * len_ratio))

print("TF saturation (doc of average length): contribution multiplier vs term count")
print(f"{'tf':>4} " + " ".join(f"k1={k:<4}" for k in (0.5, 1.2, 2.0, 5.0)))
for f in (1, 2, 3, 5, 10, 20):
    print(f"{f:>4} " + " ".join(f"{tf_part(f, k):7.2f}" for k in (0.5, 1.2, 2.0, 5.0)))

print("\nLength normalization (tf=1, k1=1.2): contribution vs doc length / avgdl")
print(f"{'len/avg':>8} " + " ".join(f"b={b:<5}" for b in (0.0, 0.5, 0.75, 1.0)))
for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
    print(f"{ratio:>8} " + " ".join(f"{tf_part(1, 1.2, ratio, b):7.2f}" for b in (0.0, 0.5, 0.75, 1.0)))

# --- Snippet 5 ---
from rank_bm25 import BM25Okapi

corpus_terms = [analyze(t) for t in FULL_TEXT]
lib = BM25Okapi(corpus_terms, k1=1.2, b=0.75)

agree = 0
for q, _, _ in QUERIES:
    qt = analyze(q)
    ours = [i for i, _ in bm25.search(qt, k=1)]
    theirs = [int(np.argmax(lib.get_scores(qt)))] if any(lib.get_scores(qt)) else []
    agree += ours == theirs
print(f"Top-1 agreement with rank_bm25: {agree}/{len(QUERIES)}")

# --- Snippet 6 ---
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype="float32")

doc_vecs = embed(FULL_TEXT)

def dense_search(q: str, k: int = 3) -> list[int]:
    return list(np.argsort(-(doc_vecs @ embed([q])[0]))[:k])

def sparse_search(q: str, k: int = 3) -> list[int]:
    return [i for i, _ in bm25.search(analyze(q), k=k)]

def evaluate(search_fn, qtype: str, k: int = 3) -> tuple[float, float]:
    rows = [(q, rel) for q, rel, t in QUERIES if t == qtype]
    r1 = np.mean([IDS.index(rel) in search_fn(q, 1) for q, rel in rows])
    rk = np.mean([IDS.index(rel) in search_fn(q, k) for q, rel in rows])
    return r1, rk

print(f"{'retriever':<8} {'query type':<11} {'R@1':>5} {'R@3':>5}")
for name, fn in [("BM25", sparse_search), ("Dense", dense_search)]:
    for qtype in ("exact", "paraphrase"):
        r1, r3 = evaluate(fn, qtype)
        print(f"{name:<8} {qtype:<11} {r1:>5.2f} {r3:>5.2f}")

print("\nPer-query top-1 (✅ correct / ❌ wrong / - nothing):")
for q, rel, t in QUERIES:
    s, d = sparse_search(q, 1), dense_search(q, 1)
    mark = lambda r: "-" if not r else ("✅" if IDS[r[0]] == rel else f"❌ {IDS[r[0]]}")
    print(f"[{t[:5]}] BM25 {mark(s):<14} Dense {mark(d):<14} {q}")

# --- Snippet 7 ---
import difflib

VOCAB = list(bm25.postings)

def fix_typos(terms: list[str]) -> list[str]:
    """Map unknown query terms to the closest vocabulary term (fuzzy matching)."""
    fixed = []
    for t in terms:
        if t in bm25.postings or any(ch.isdigit() for ch in t):
            fixed.append(t)
        else:
            match = difflib.get_close_matches(t, VOCAB, n=1, cutoff=0.8)
            fixed.append(match[0] if match else t)
    return fixed

def phrase_filter(results: list[tuple[int, float]], phrase: str) -> list[tuple[int, float]]:
    return [(i, s) for i, s in results if phrase.lower() in FULL_TEXT[i].lower()]

# 1) Typos
typo_q = "procesing fee on hom loan"
print("Typo query terms :", analyze(typo_q))
print("Fixed terms       :", fix_typos(analyze(typo_q)))
print("Without fix       :", [(IDS[i], round(s, 2)) for i, s in bm25.search(analyze(typo_q))])
print("With fix          :", [(IDS[i], round(s, 2)) for i, s in bm25.search(fix_typos(analyze(typo_q)))])

# 2) Phrase: all words present is not enough, they must be adjacent and in order
q = "minimum balance"
cands = bm25.search(analyze(q), k=5)
print("\nBag-of-words     :", [IDS[i] for i, _ in cands])
print("Phrase-filtered  :", [IDS[i] for i, _ in phrase_filter(cands, "minimum balance")])

# 3) Field boosting (BM25F-lite): the title counts double
title_bm25 = BM25([analyze(title) for _, title, _ in DOCS])
body_bm25 = BM25([analyze(text) for _, _, text in DOCS])

def fielded_search(q: str, w_title: float = 2.0, k: int = 3):
    qt = analyze(q)
    scores = defaultdict(float)
    for i, s in title_bm25.search(qt, k=len(DOCS)):
        scores[i] += w_title * s
    for i, s in body_bm25.search(qt, k=len(DOCS)):
        scores[i] += s
    return sorted(scores.items(), key=lambda x: -x[1])[:k]

q = "home loan tax"
print("\nNo field boost   :", [(IDS[i], round(s, 2)) for i, s in bm25.search(analyze(q))])
print("Title boost x2   :", [(IDS[i], round(s, 2)) for i, s in fielded_search(q)])

# --- Snippet 8 ---
import sqlite3

con = sqlite3.connect(":memory:")
con.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, title, body, tokenize='porter unicode61')")
con.executemany("INSERT INTO docs (doc_id, title, body) VALUES (?, ?, ?)", DOCS)

def fts_search(question: str, k: int = 3, title_weight: float = 2.0):
    terms = [t for t in re.findall(r"[A-Za-z0-9]+", question) if t.lower() not in STOPWORDS]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)              # quote terms -> no FTS syntax injection
    sql = f"""SELECT doc_id, bm25(docs, {title_weight}, 1.0) AS score
              FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?"""  # bm25(): LOWER is better
    return [(d, round(-s, 2)) for d, s in con.execute(sql, (match, k))]

for q in ["What is the IFSC code of the Pune branch?", "What does UPI error E1043 mean?",
          "When should I submit Form 15H?", "I misplaced my plastic, how do I freeze it?"]:
    print(f"{q:<48} -> {fts_search(q)}")

# --- Snippet 9 ---
# pip install transformers torch
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

name = "naver/splade-cocondenser-ensembledistil"
tok = AutoTokenizer.from_pretrained(name)
mlm = AutoModelForMaskedLM.from_pretrained(name).eval()

def splade_terms(text: str, top: int = 15) -> list[tuple[str, float]]:
    inputs = tok(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = mlm(**inputs).logits                           # [1, seq_len, vocab]
    weights = torch.max(torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1), dim=1).values[0]
    idx = torch.nonzero(weights).squeeze(1).tolist()
    pairs = [(tok.convert_ids_to_tokens(i), float(weights[i])) for i in idx]
    return sorted(pairs, key=lambda x: -x[1])[:top]

print(splade_terms("cost of borrowing to buy a house"))
print(splade_terms("block my lost card"))

if __name__ == "__main__":
    print("Starting Day 12 script execution...")
