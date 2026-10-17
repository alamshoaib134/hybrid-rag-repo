# Day 08: Document Loaders (PDF, HTML, DOCX, Tables)

**Chapter**: Search & Context  
**Concept**: Extracting clean text and tables from messy files

---

## 📌 Cheat Sheet & Quick Summary

| Concept | Remember this |
|---|---|
| **Why it matters** | Retrieval can only find what parsing **extracted correctly**. Many "RAG quality" bugs are really **parsing bugs**. |
| **PDF reality** | A PDF stores **positioned glyphs**, not paragraphs. Columns, tables, headers/footers, and hyphenation must be **reconstructed**. |
| **Digital vs scanned** | Digital PDF -> text extraction. Scanned PDF (≈ no extractable text) -> **OCR** or a **vision-based parser**. |
| **PDF tools** | `pypdf` (simple), `pdfplumber` (tables, layout), PyMuPDF/`pymupdf4llm` (fast, Markdown; **AGPL**), Docling / Unstructured / Marker / LlamaParse (layout-aware), cloud OCR. |
| **HTML** | Remove boilerplate (nav, footer, cookie banners, scripts). Keep **headings** and **tables**. JS-heavy pages need a headless browser. |
| **DOCX** | Read blocks **in document order**. Heading styles -> **section path** metadata. Tables -> Markdown. |
| **Tables** | Serialize as **Markdown** (whole table) or **row sentences** ("Product: FD; Rate: 7.1%"). Very large or numeric tables -> **SQL** (Day 24). |
| **Clean** | Unicode NFKC, de-hyphenate line breaks, strip repeated headers/footers and page numbers, collapse whitespace, dedup. |
| **Metadata** | `source`, `page`, `section`, `doc_type`, `title`, `effective_date`, `department`, `access_level`, `parser`. |
| **Verify** | A **fact-survival test**: key facts (for example, "7.9%") must survive parsing; noise ("CONFIDENTIAL", "Accept cookies") must not. |

---
## 📂 Contents & Structure

- [`day_08.ipynb`](./day_08.ipynb): Interactive Jupyter Notebook containing the full theoretical explanation, math derivations, architectural illustrations, and step-by-step runnable code cells.
- [`main.py`](./main.py): Standalone Python script consolidating the day's algorithms and test cases.

## 🚀 How to Run

### 1. Interactive Notebook (Recommended)
Launch Jupyter from the repository root:
```bash
jupyter notebook day_08_document_loaders/day_08.ipynb
```

### 2. Standalone Script
Run directly from your terminal:
```bash
python day_08_document_loaders/main.py
```
