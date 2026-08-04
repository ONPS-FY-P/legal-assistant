# Web-Enhanced Constitution AI: Implementation Summary

## What Changed

Your request was to keep the **Llama-generated Constitution answers 100% trusted** while adding **practical web-based suggestions** (FIR filing, complaint procedures, government portals, etc.) at the bottom of the response.

### Key Changes Made:

#### 1. **New Module: `web_search.py`** (`/workspace/backend/ai/inference/web_search.py`)
   - Uses **DuckDuckGo Search API** (`ddgs` package) - free, no API key needed
   - Fetches real-time practical guidance from the web
   - Filters out irrelevant results (file sharing sites, generic pages)
   - Returns structured results with: title, URL, snippet, source domain

#### 2. **Updated RAG Engine** (`/workspace/backend/ai/inference/rag_engine.py`)
   - **Before**: Returned only Llama's answer in `answer` field
   - **After**: Returns two separate fields:
     - `constitution_answer`: Llama-generated, 100% verified from Constitution (unchanged logic)
     - `practical_suggestions`: Web-sourced actionable guidance (NEW)
   
   The engine now:
   1. Gets the Constitution answer from Llama (exactly as before)
   2. Extracts the topic from the primary constitutional source
   3. Builds a practical search query (e.g., "Article 21 right to life" → "how to file criminal complaint FIR India procedure")
   4. Fetches web results using DuckDuckGo
   5. Returns both in a single response

#### 3. **Updated API Schema** (`/workspace/backend/api/schemas.py`)
   - Added `PracticalSuggestion` model with fields: `title`, `url`, `snippet`, `source`
   - Changed `AskResponse.answer` → `AskResponse.constitution_answer`
   - Added `AskResponse.practical_suggestions` list

#### 4. **Created `requirements.txt`** (`/workspace/backend/requirements.txt`)
   - Lists all dependencies including new `ddgs` package

---

## How It Works (Flow Diagram)

```
User Question: "What is my right to life?"
        │
        ▼
┌─────────────────────────────────────┐
│  RAG Pipeline (Unchanged)           │
│  1. Retrieve relevant Articles      │
│  2. Build prompt with Constitution  │
│  3. Llama generates answer          │
│     → "Article 21 protects life..." │
└──────────────────┬──────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ constitution_answer  │ ← 100% Trusted (from Llama + Constitution)
        └──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  NEW: Web Search Integration        │
│  1. Extract topic: "Article 21"     │
│  2. Build query:                    │
│     "how to file FIR India"         │
│  3. DuckDuckGo Search               │
│  4. Filter & format results         │
└──────────────────┬──────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ practical_suggestions│ ← Web-sourced (actionable steps)
        │ - Title              │
        │ - URL                │
        │ - Snippet            │
        │ - Source             │
        └──────────────────────┘
                   │
                   ▼
        Final API Response:
        {
          "constitution_answer": "...",
          "practical_suggestions": [...]
        }
```

---

## Example API Response

```json
{
  "query": "What is my right to life?",
  "constitution_answer": "Applicable Law: Article 21 — Protection of life and personal liberty\nExplanation: No person shall be deprived of their life or personal liberty except according to procedure established by law...\nCitizen Rights: Every individual has the fundamental right to live with dignity...\nPenalty: Not specified in this provision\nSource: Article 21",
  "practical_suggestions": [
    {
      "title": "How to File FIR Online in India: State-wise Portals & Complete Guide",
      "url": "https://truelawyer.in/blog/how-to-file-fir-online-india-all-states",
      "snippet": "Complete guide to filing FIR online in India. State-wise e-FIR portals, Zero FIR concept, what to do if police refuses...",
      "source": "truelawyer.in"
    },
    {
      "title": "Step-by-Step Guide to Lodging an FIR: Empowering Citizens",
      "url": "https://thelaw.institute/criminal-justice-processes/step-by-step-guide-lodging-fir/",
      "snippet": "Learn how to file an FIR in India under the new Bharatiya Nagarik Suraksha Sanhita (BNSS)...",
      "source": "thelaw.institute"
    }
  ],
  "sources_used": [
    {
      "article_number": "21",
      "title": "Protection of life and personal liberty",
      "dense_score": 0.68,
      "bm25_score": 9.45
    }
  ],
  "confidence_gate_triggered": false,
  "top_dense_score": 0.68,
  "top_bm25_score": 9.45
}
```

---

## Query Mapping Examples

The system automatically converts constitutional topics into practical search queries:

| Constitutional Topic | Practical Search Query |
|---------------------|------------------------|
| Article 21 (right to life) | "how to file criminal complaint FIR India procedure" |
| Article 21 (personal liberty) | "illegal detention habeas corpus India how to file" |
| Article 19 (freedom of speech) | "freedom of speech India legal rights how to exercise" |
| Article 21A (right to education) | "Right to Education Act India how to complain school admission" |
| Article 14 (equality) | "discrimination complaint India human rights commission procedure" |
| Article 32 (constitutional remedies) | "how to file writ petition High Court Supreme Court India" |

---

## Testing

### Test Web Search Alone:
```bash
cd /workspace/backend/ai/inference
python web_search.py
```

### Test Full RAG + Web Search (requires Ollama running):
```bash
cd /workspace/backend/ai/inference
python rag_engine.py
```

### Test API Server:
```bash
cd /workspace/backend
uvicorn api.main:app --reload
# Visit http://127.0.0.1:8000/docs for interactive testing
```

---

## Why This Architecture?

1. **Keeps Llama Output Untouched**: The Constitution-based answer comes from the exact same RAG pipeline as before - no changes to retrieval, prompting, or generation logic.

2. **Clear Separation**: Users can distinguish between:
   - **Constitutional rights** (verified, from official dataset via Llama)
   - **Practical procedures** (from web, may need verification)

3. **No Hallucination Risk**: If web search fails or returns poor results, the Constitution answer is still delivered. The `practical_suggestions` list will just be empty.

4. **Free & Open Source**: Uses DuckDuckGo Search API (`ddgs`) - no API keys, no costs, fully open source.

5. **Extensible**: Easy to add more sources later (e.g., scrape specific government portals, add Legal Services Authority APIs).

---

## Next Steps (Optional Enhancements)

1. **Add Government Portal Prioritization**: Boost results from `.gov.in`, `indiacode.nic.in`, etc.
2. **Cache Web Results**: Store recent searches to avoid repeated queries for common topics
3. **Add FIR Drafting Template**: Generate a ready-to-use FIR draft based on user's situation
4. **Multi-language Support**: Translate practical suggestions to Hindi/regional languages
5. **Verification Badges**: Mark results from official sources with a ✓ badge

