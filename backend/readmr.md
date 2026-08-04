# Constitution AI

A **RAG-based legal assistant** for Indian citizens, built as a Final Year Project. It answers questions about the Constitution of India using retrieval-augmented generation — grounding every answer in real constitutional text rather than an LLM's memory, so responses are traceable to an actual Article or Schedule.

> Started with the Constitution. Motor Vehicles Act, BNS, BNSS, and BSA are planned next (see Roadmap).

---

## Why RAG, not a fine-tuned/trained model

This was a deliberate architecture decision, not a default. A legal assistant's core requirement is **factual accuracy with traceable sources** — training or fine-tuning an LLM bakes facts into opaque weights, where hallucination is undetectable and updating the law means retraining. RAG instead retrieves the actual legal text and forces the model to answer only from it, so every answer can be checked against its source. See the project's build log for the full comparison.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Mobile frontend | React Native (Expo) — *planned, Sprint 3* |
| Backend API | FastAPI |
| LLM | Ollama running Llama 3.2 (fully local, no API costs) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector search | FAISS (`IndexFlatIP`, cosine similarity via normalized vectors) |
| Keyword search | BM25 (`rank_bm25`) — combined with FAISS via hybrid retrieval |
| Dataset source | Official Constitution of India PDF (Ministry of Law and Justice) |

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────┐
│              Retriever                   │
│  ┌───────────────┐   ┌────────────────┐ │
│  │ FAISS (dense)  │   │ BM25 (keyword) │ │
│  │ MiniLM vectors │   │ exact terms    │ │
│  └───────┬────────┘   └───────┬────────┘ │
│          └─────────┬──────────┘          │
│                     ▼                     │
│         Reciprocal Rank Fusion (RRF)      │
└─────────────────────┬─────────────────────┘
                       ▼
           Confidence Gate (dense OR bm25
           score must clear threshold)
                       │
        ┌──────────────┴──────────────┐
        ▼                              ▼
  Below threshold               Above threshold
        │                              │
        ▼                              ▼
"I don't have enough          Prompt Builder
 information..."              (5-field template:
                               Applicable Law,
                               Explanation, Citizen
                               Rights, Penalty, Source)
                                       │
                                       ▼
                               Ollama (Llama 3.2)
                                       │
                                       ▼
                               Structured Answer
```

**Why hybrid retrieval:** pure dense (FAISS/MiniLM) search is sensitive to exact phrasing — short, casual questions like "What is my right to life?" scored below a pure-similarity threshold even though Article 21 is the obviously correct answer, because it literally contains the words "life" and "liberty." BM25 keyword search catches this reliably. Reciprocal Rank Fusion combines both rankings without needing to normalize their differently-scaled scores.

**Why the confidence gate:** if neither retrieval method finds a strong match, the system explicitly refuses to answer rather than letting Ollama guess — this is what prevents hallucinated Article numbers on out-of-scope questions (e.g. Motor Vehicles Act topics not yet in the dataset).

**Why chunk framing matters:** Schedule chunks (unlike Articles) are often bare lists/tables in the source PDF with no narrative sentence. Prepending a declarative framing sentence per Schedule (e.g. *"The Eighth Schedule of the Constitution of India lists the following officially recognized languages of India: ..."*) measurably improved both retrieval scores and the LLM's citation accuracy — see the dataset section below.

---

## Folder Structure

```
backend/
├── ai/
│   ├── ingestion/                  # OFFLINE pipeline — run manually when the dataset changes
│   │   ├── document_loader.py      # PDF → raw text per page
│   │   ├── cleaner.py              # strips Hindi/Devanagari, normalizes whitespace
│   │   ├── embeddings.py           # generates MiniLM embeddings for all chunks
│   │   └── vector_store.py         # builds/saves/loads the FAISS index
│   └── inference/                  # ONLINE pipeline — runs per API request
│       ├── retriever.py            # hybrid FAISS + BM25 search with RRF fusion
│       ├── prompt_builder.py       # builds the grounded, structured prompt
│       ├── ollama_client.py        # talks to the local Ollama server
│       └── rag_engine.py           # orchestrates retrieval → gate → prompt → generation
├── datasets/
│   ├── raw/
│   │   └── constitution_of_india.pdf
│   ├── processed/
│   │   └── constitution_full_dataset.json   # 699 chunks: 579 Article + 120 Schedule
│   └── vector_db/
│       ├── embeddings.npy
│       ├── metadata.json
│       └── index.faiss
├── api/
│   ├── main.py                     # FastAPI app, loads RAGEngine once at startup
│   └── schemas.py                  # Pydantic request/response models
└── requirements.txt

frontend/                            # planned — Sprint 3
```

---

## Dataset

Built from the official Constitution of India PDF (402 pages, diglot Hindi/English edition) through a custom extraction pipeline:

1. Extracted all pages, filtered to English-only text (Devanagari stripped — multilingual support is future scope)
2. Skipped front matter/Table of Contents, located the real Article body text (starts at PDF page 33)
3. Parsed all **493 Articles** (1–395, including lettered insertions like 21A, 244A, 371A–J, and genuinely omitted articles) using a monotonic-numbering parser that correctly distinguishes real article headers from amendment-citation footnotes — a non-trivial problem, since footnotes like *"3. Ins. by the Constitution (Thirty-fifth Amendment) Act, 1974..."* look identical to article headers at a glance
4. Parsed all **12 Schedules** (state/UT lists, oaths, Union/State/Concurrent Lists, official languages, anti-defection law, etc.), tagged with their internal sub-part structure (e.g. Seventh Schedule → `List I / II / III`)
5. Sub-chunked long articles/schedules (>1800 characters) at clause boundaries so no legal sentence is cut mid-way

**Result:** `constitution_full_dataset.json` — 715 chunks total (579 from Articles, 136 from Schedules), each carrying metadata (`act`, `article_number` or `schedule_name`, `part`, `title`).

**A worthwhile debugging lesson from this project:** an early version of the Eighth Schedule (Official Languages) chunk was a bare list — `"Articles 344(1) and 351 Languages 1. Assamese. 2. Bengali..."` — with leaked page-header noise (`"326 THE CONSTITUTION OF INDIA"`) and no framing sentence. The LLM consistently cited Article 351 instead of the Eighth Schedule for language-related questions, even with strict prompt instructions telling it to prefer the top-ranked source. The fix wasn't prompt engineering — it was rewriting the chunk with a declarative opening sentence (`"The Eighth Schedule of the Constitution of India lists the following officially recognized languages of India: ..."`) and stripping the page-header noise. This raised the chunk's retrieval score from 0.60 to 0.78 (dense) and 8.0 to 22.5 (BM25), and the LLM's citation immediately became correct. **What looked like a prompt problem was a data quality problem** — worth checking retrieved chunk text directly before assuming the LLM "isn't listening."

---

## Setup

```powershell
# 1. Clone/create project, set up virtual environment
cd backend
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the Constitution PDF
#    → datasets/raw/constitution_of_india.pdf

# 4. Pull the Ollama model (Ollama must be installed and running)
ollama pull llama3.2

# 5. Build the dataset (or use the pre-built constitution_full_dataset.json)
#    → datasets/processed/constitution_full_dataset.json

# 6. Generate embeddings and build the FAISS index
python ai/ingestion/embeddings.py
python ai/ingestion/vector_store.py

# 7. Run the API
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API testing.

---

## API

### `GET /health`
Returns index size and Ollama availability.
```json
{ "status": "ok", "chunks_indexed": 699, "ollama_available": true }
```

### `POST /ask`
```json
// Request
{ "query": "What is my right to life?" }

// Response
{
  "query": "What is my right to life?",
  "answer": "Applicable Law: Article 21 — Protection of life and personal liberty\nExplanation: ...\nCitizen Rights: ...\nPenalty: Not specified in this provision\nSource: Article 21",
  "sources_used": [
    { "article_number": "21", "title": "Protection of life and personal liberty", "dense_score": 0.42, "bm25_score": 9.18 }
  ],
  "confidence_gate_triggered": false,
  "top_dense_score": 0.42,
  "top_bm25_score": 9.18
}
```

---

## Progress (Sprint 1 — Complete)

- [x] Constitution PDF → cleaned, chunked dataset (Articles + Schedules)
- [x] MiniLM embeddings + FAISS index
- [x] Hybrid retrieval (FAISS + BM25 via RRF)
- [x] Confidence gate to prevent hallucinated answers on out-of-scope queries
- [x] Ollama (Llama 3.2) integration with structured 5-field prompt template
- [x] FastAPI backend with `/ask` and `/health` endpoints

## Roadmap

- **Sprint 2:** Expand dataset — Motor Vehicles Act, BNS, BNSS, BSA
- **Sprint 3:** React Native (Expo) mobile app, voice input/output
- **Sprint 4:** Testing (RAG quality eval set), deployment
- **Future scope:** FIR/complaint drafting assistant, OCR, legal knowledge graph, court judgment retrieval, offline AI, multilingual support

## Known Limitations

- Dense retrieval alone is sensitive to short/casual phrasing — mitigated, not eliminated, by hybrid BM25 fusion.
- Schedules dataset relies on heuristic sub-heading detection (e.g. "PART A", "LIST I") — mostly reliable but not manually verified line-by-line against the source PDF. Two chunks (in the Sixth Schedule) retain minor leftover Gazette/amendment-citation text.
- The full RAG pipeline has been manually tested on a handful of queries, not a systematic evaluation set — Sprint 4's Testing phase should build a golden Q&A set to measure retrieval precision and answer accuracy properly.