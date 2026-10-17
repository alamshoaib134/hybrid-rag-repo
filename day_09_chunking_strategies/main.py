"""
Day 09: Chunking Strategies (Fixed, Recursive, Semantic)
Core Concept: Breaking documents into optimal context units
Chapter: Search & Context

This script executes the key algorithmic implementations and examples from Day 9.
"""
from __future__ import annotations
import os
import sys

def main():
    print("=" * 60)
    print("Executing Day 09: Chunking Strategies (Fixed, Recursive, Semantic)")
    print("=" * 60)

# --- Snippet 1 ---
import re
from dataclasses import dataclass, field

import numpy as np
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def ntok(text: str) -> int:
  return len(enc.encode(text))


@dataclass
class Chunk:
  text: str
  metadata: dict = field(default_factory=dict)


HANDBOOK = """# ZyntraCorp Retail Banking Handbook 2026

This handbook summarises the key retail banking products of ZyntraCorp Bank for the financial year 2026. It is intended for customers and branch staff. Where this handbook conflicts with a signed loan agreement, the agreement prevails.

## 1. Home Loans

ZyntraCorp offers home loans for purchasing, constructing, or renovating residential property in India. Loans are available on floating and fixed interest rates.

### 1.1 Interest Rates

Floating home loan interest rates start at 7.9% per annum for salaried customers. Self-employed customers are offered rates starting at 8.3% per annum. The final rate depends on the applicant's credit score, loan amount, and loan-to-value ratio. Rates are reset every quarter based on the bank's external benchmark.

### 1.2 Fees and Charges

A processing fee of 0.5% of the loan amount is charged, capped at 10,000 rupees. There are no prepayment charges on floating rate loans for individual borrowers. Fixed rate loans attract a prepayment charge of 2% of the prepaid amount. Late EMI payments attract a penalty of 2% per month on the overdue amount.

### 1.3 Tenure, Eligibility and Documents

The maximum tenure is 30 years, or until the borrower turns 70, whichever is earlier. Applicants must be between 21 and 65 years old at the time of application. Salaried applicants must submit the last three salary slips, six months of bank statements, and Form 16. Self-employed applicants must submit two years of income tax returns and audited financial statements.

## 2. Fixed Deposits

Fixed deposits (FDs) can be opened online or at any branch with a minimum amount of 5,000 rupees. Interest can be paid out monthly or reinvested until maturity.

### 2.1 Standard Rates

The table below lists standard FD rates for resident individuals.

| Tenure | Rate (per annum) |
|---|---|
| 7 days to 1 year | 6.2% |
| 1 to 2 years | 7.1% |
| 2 to 5 years | 6.9% |
| 5 to 10 years | 6.5% |

### 2.2 Senior Citizens

Customers aged 60 and above receive an additional benefit. Customers in this category earn 7.4% on deposits of 1 to 2 years, and 0.5 percentage points above the standard rate for other tenures.

### 2.3 Premature Withdrawal

Deposits can be withdrawn before maturity. A penalty of 1.25% is deducted from the applicable interest rate. No interest is paid on deposits withdrawn within seven days of opening.

## 3. Savings Accounts

The Savings Plus account is our flagship savings product. It offers 4.25% annual interest, credited quarterly, on the daily closing balance. There is no minimum balance requirement. UPI transfers are free and unlimited. The first debit card is issued free of charge; replacement cards cost 200 rupees.

## 4. Credit Cards

### 4.1 Platinum Card

The Platinum credit card carries an annual fee of 2,500 rupees. It is waived if total spends in the previous card year exceed 3 lakh rupees. Cardholders earn 2% cashback on online purchases and 1% on all other purchases. The card includes four complimentary airport lounge visits per year.

### 4.2 Lost or Stolen Cards

If your card is lost or stolen, open the mobile app, go to Cards, and tap Block Card. The card is blocked instantly and a replacement is dispatched within five working days. You are not liable for fraudulent transactions reported within three days.

## 5. Customer Support and Grievances

Customer support is available 24/7 by phone at 1800-000-000 and through in-app chat. Complaints can be raised through the app, by email, or at any branch. The bank aims to resolve every complaint within 15 working days. If a complaint is not resolved within 30 days, or the customer is not satisfied with the response, the customer may escalate it to the Banking Ombudsman under the Reserve Bank's Integrated Ombudsman Scheme.
"""

# (question, answer string that must appear in a retrieved chunk)
EVAL = [
    (
        "What is the home loan interest rate for salaried customers?",
        "7.9%",
    ),
    ("What is the processing fee on a home loan?", "0.5%"),
    (
        "Are there prepayment charges on floating rate home loans?",
        "no prepayment charges",
    ),
    ("What is the maximum home loan tenure?", "30 years"),
    (
        "What documents does a salaried person need for a home loan?",
        "salary slips",
    ),
    ("What is the FD rate for senior citizens?", "7.4%"),
    ("What is the FD rate for 2 to 5 years?", "6.9%"),
    (
        "What is the penalty for breaking a fixed deposit early?",
        "1.25%",
    ),
    (
        "Do I need to maintain a minimum balance in Savings Plus?",
        "no minimum balance",
    ),
    ("How often is savings interest credited?", "quarterly"),
    ("What is the Platinum card annual fee?", "2,500 rupees"),
    ("When is the Platinum card fee waived?", "3 lakh"),
    (
        "How many lounge visits does the Platinum card give?",
        "four complimentary",
    ),
    ("How do I block a lost card?", "Block Card"),
    ("What is the customer care phone number?", "1800-000-000"),
    (
        "How long does the bank take to resolve complaints?",
        "15 working days",
    ),
    (
        "Where can I escalate an unresolved complaint?",
        "Banking Ombudsman",
    ),
]


def chunk_stats(chunks: list[Chunk], name: str) -> None:
  sizes = [ntok(c.text) for c in chunks]
  last_lines = [c.text.strip().splitlines()[-1].strip() for c in chunks]
  dangling = sum(1 for ln in last_lines if ln.startswith("#"))
  mid = sum(
      1
      for ln in last_lines
      if not ln.startswith("#") and not re.search(r"[.!?]$", ln)
  )
  print(
      f"{name:<18} n={len(chunks):>3} avg={np.mean(sizes):6.1f}"
      f" min={min(sizes):>4} max={max(sizes):>4} "
      f"mid_sentence={mid / len(chunks):4.0%} "
      f"dangling_heading={dangling / len(chunks):4.0%}"
  )


print("Handbook tokens:", ntok(HANDBOOK))

# --- Snippet 2 ---
def fixed_token_chunks(
    text: str, size: int = 100, overlap: int = 20
) -> list[Chunk]:
  ids = enc.encode(text)
  step = size - overlap
  chunks = []
  for i, start in enumerate(range(0, len(ids), step)):
    chunks.append(
        Chunk(
            enc.decode(ids[start : start + size]),
            {"strategy": "fixed", "idx": i},
        )
    )
    if start + size >= len(ids):
      break
  return chunks


fixed = fixed_token_chunks(HANDBOOK, size=100, overlap=20)
chunk_stats(fixed, "fixed(100, 20)")
for c in fixed[4:6]:
  print("\n---", c.metadata, "---\n", c.text)

# --- Snippet 3 ---
SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_sentences(text: str) -> list[str]:
  """Sentences for prose; headings, list items, and whole tables are kept as single units."""
  units = []
  for block in re.split(r"\n\s*\n", text):
    block = block.strip()
    if not block:
      continue
    if block.startswith("|"):  # a table stays whole
      units.append(block)
    elif block.startswith(("#", "-", "* ")):
      units.extend(ln.strip() for ln in block.splitlines() if ln.strip())
    else:
      units.extend(
          SENT_BOUNDARY.split(block.replace("\n", " "))
      )
  return [u.strip() for u in units if u.strip()]


def sentence_chunks(
    text: str, max_tokens: int = 100, overlap_sentences: int = 1
) -> list[Chunk]:
  chunks, cur = [], []
  for s in split_sentences(text):
    if cur and ntok(" ".join(cur + [s])) > max_tokens:
      chunks.append(" ".join(cur))
      cur = cur[-overlap_sentences:] if overlap_sentences else []
    cur.append(s)
  if cur:
    chunks.append(" ".join(cur))
  return [
      Chunk(t, {"strategy": "sentence", "idx": i})
      for i, t in enumerate(chunks)
  ]


sent = sentence_chunks(HANDBOOK, max_tokens=100)
chunk_stats(sent, "sentence(100)")
print("\n", sent[5].text)

# --- Snippet 4 ---
def recursive_split(
    text: str,
    max_tokens: int = 100,
    separators: tuple = ("\n\n", "\n", ". ", " "),
) -> list[str]:
  if ntok(text) <= max_tokens:
    return [text.strip()] if text.strip() else []
  sep = next((s for s in separators if s in text), None)
  if sep is None:  # last resort: hard token cut
    ids = enc.encode(text)
    return [
        enc.decode(ids[i : i + max_tokens])
        for i in range(0, len(ids), max_tokens)
    ]
  parts = text.split(sep)
  pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
  finer = separators[separators.index(sep) + 1 :]
  out, cur = [], ""
  for p in pieces:
    if ntok(p) > max_tokens:  # piece too big: recurse with finer separators
      if cur.strip():
        out.append(cur.strip())
      cur = ""
      out.extend(recursive_split(p, max_tokens, finer))
    elif ntok(cur + p) <= max_tokens:  # greedily merge small pieces
      cur += p
    else:
      if cur.strip():
        out.append(cur.strip())
      cur = p
  if cur.strip():
    out.append(cur.strip())
  return out


rec = [
    Chunk(t, {"strategy": "recursive", "idx": i})
    for i, t in enumerate(recursive_split(HANDBOOK, 100))
]
chunk_stats(rec, "recursive(100)")
for c in rec[:3]:
  print("\n---\n", c.text)

# --- Snippet 5 ---
def markdown_sections(text: str) -> list[tuple[str, str]]:
  """Return (section_path, body) pairs, using Markdown headings."""
  sections, path, buf = [], [], []

  def flush():
    body = "\n".join(buf).strip()
    if body:
      sections.append((" > ".join(path), body))
    buf.clear()

  for line in text.splitlines():
    m = re.match(r"^(#{1,6})\s+(.*)", line)
    if m:
      flush()
      level = len(m.group(1))
      path[:] = path[: level - 1] + [m.group(2).strip()]
    else:
      buf.append(line)
  flush()
  return sections


def structure_chunks(
    text: str, max_tokens: int = 150
) -> list[Chunk]:
  chunks = []
  for section, body in markdown_sections(text):
    pieces = []
    for block in [
        b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()
    ]:
      if block.startswith("|") or ntok(block) <= max_tokens:
        pieces.append(block)  # tables and small paragraphs stay whole
      else:
        pieces.extend(recursive_split(block, max_tokens))
    cur = ""
    for p in pieces:  # pack blocks within the section
      if cur and ntok(cur + "\n\n" + p) > max_tokens:
        chunks.append((section, cur))
        cur = p
      else:
        cur = f"{cur}\n\n{p}" if cur else p
    if cur:
      chunks.append((section, cur))
  return [
      Chunk(
          f"[{sec}]\n{body}",
          {"strategy": "structure", "section": sec, "idx": i},
      )
      for i, (sec, body) in enumerate(chunks)
  ]


struct = structure_chunks(HANDBOOK, max_tokens=150)
chunk_stats(struct, "structure(150)")
for c in struct:
  if "Senior" in c.metadata["section"]:
    print("\n", c.text)

# --- Snippet 6 ---
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(texts: list[str]) -> np.ndarray:
  return np.asarray(
      _model.encode(texts, normalize_embeddings=True), dtype="float32"
  )


def semantic_chunks(
    text: str,
    percentile: float = 80,
    window: int = 1,
    max_tokens: int = 200,
):
  sents = [
      s
      for s in split_sentences(text)
      if not s.startswith("#")
  ]  # headings aren't sentences
  windows = [
      " ".join(sents[max(0, i - window) : i + window + 1])
      for i in range(len(sents))
  ]
  vecs = embed(windows)
  dists = np.array(
      [1 - float(vecs[i] @ vecs[i + 1]) for i in range(len(vecs) - 1)]
  )
  threshold = float(np.percentile(dists, percentile))

  chunks, cur = [], [sents[0]]
  for i, d in enumerate(dists):
    nxt = sents[i + 1]
    if (
        d > threshold
        or ntok(" ".join(cur + [nxt])) > max_tokens
    ):
      chunks.append(" ".join(cur))
      cur = [nxt]
    else:
      cur.append(nxt)
  chunks.append(" ".join(cur))
  return [
      Chunk(
          t,
          {
              "strategy": "semantic",
              "idx": i,
          },
      )
      for i, t in enumerate(chunks)
  ], sents, dists, threshold


sem, sents, dists, thr = semantic_chunks(
    HANDBOOK, percentile=80
)
chunk_stats(sem, "semantic(p80)")

print(f"\nBreakpoint threshold (80th percentile distance): {thr:.3f}")
for i, d in enumerate(dists[:14]):
  mark = " <-- BREAK" if d > thr else ""
  print(f"{d:.3f} | {sents[i][:55]:<55} -> {sents[i + 1][:30]}{mark}")

# --- Snippet 7 ---
def evaluate_chunks(
    chunks: list[Chunk], k: int = 3
) -> dict:
  vecs = embed([c.text for c in chunks])
  hits, ctx_tokens = 0, 0
  for question, answer in EVAL:
    top = np.argsort(-(vecs @ embed([question])[0]))[:k]
    context = [chunks[i].text for i in top]
    ctx_tokens += sum(ntok(t) for t in context)
    hits += any(answer.lower() in t.lower() for t in context)
  return {
      "recall": hits / len(EVAL),
      "ctx_tokens": ctx_tokens / len(EVAL),
  }


# Sanity check: every answer string really exists in the handbook
assert all(a.lower() in HANDBOOK.lower() for _, a in EVAL)

strategies = {
    "fixed(100,20)": fixed,
    "sentence(100)": sent,
    "recursive(100)": rec,
    "structure(150)": struct,
    "semantic(p80)": sem,
}
print(
    f"{'strategy':<16} {'chunks':>6} {'avg tok':>8} {'R@1':>6}"
    f" {'R@3':>6} {'ctx tok@3':>10}"
)
for name, chunks in strategies.items():
  r1, r3 = evaluate_chunks(chunks, k=1), evaluate_chunks(chunks, k=3)
  avg = np.mean([ntok(c.text) for c in chunks])
  print(
      f"{name:<16} {len(chunks):>6} {avg:>8.1f}"
      f" {r1['recall']:>6.2f} {r3['recall']:>6.2f}"
      f" {r3['ctx_tokens']:>10.0f}"
  )

# --- Snippet 8 ---
print(f"{'strategy':<12} {'size':>5} {'chunks':>6} {'R@3':>6} {'ctx tok@3':>10}")
for size in (50, 100, 200, 400):
  for name, chunker in [
      (
          "recursive",
          lambda s: [
              Chunk(t) for t in recursive_split(HANDBOOK, s)
          ],
      ),
      (
          "structure",
          lambda s: structure_chunks(HANDBOOK, s),
      ),
  ]:
    chunks = chunker(size)
    r = evaluate_chunks(chunks, k=3)
    print(
        f"{name:<12} {size:>5} {len(chunks):>6}"
        f" {r['recall']:>6.2f} {r['ctx_tokens']:>10.0f}"
    )

# --- Snippet 9 ---
# LangChain: header-aware split, then token-based recursive split inside sections
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
)
sections = md_splitter.split_text(HANDBOOK)
token_splitter = (
    RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=150,
        chunk_overlap=20,
    )
)
lc_chunks = token_splitter.split_documents(sections)
print("LangChain chunks:", len(lc_chunks))
print(
    lc_chunks[6].metadata,
    "\n",
    lc_chunks[6].page_content[:200],
)

# --- Snippet 10 ---
# LlamaIndex: Markdown-aware node parser (and a semantic splitter)
from llama_index.core import Document
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
)

md_nodes = MarkdownNodeParser().get_nodes_from_documents(
    [Document(text=HANDBOOK)]
)
print("LlamaIndex markdown nodes:", len(md_nodes))
print(
    md_nodes[6].metadata,
    "\n",
    md_nodes[6].get_content()[:200],
)

# Semantic splitter (needs a real embedding model, e.g., HuggingFaceEmbedding from Day 7):
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=90,
#                                       embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"))
# sem_nodes = splitter.get_nodes_from_documents([Document(text=HANDBOOK)])

if __name__ == "__main__":
    print("Starting Day 09 script execution...")
