"""
main.py

Purpose (SDD Section 6 -- API Design):
- The FastAPI app itself. Exposes /ask (the core RAG endpoint), /health,
  and / (root).
- RAGEngine is loaded ONCE at startup (via FastAPI's lifespan), not
  per-request -- this is what makes /ask fast after the first request.
  Without this, every question would reload MiniLM + FAISS + re-check
  Ollama, adding seconds of dead latency to every call.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# rag_engine.py lives in ai/inference -- add it to the import path
_INFERENCE_DIR = Path(__file__).resolve().parent.parent / "ai" / "inference"
sys.path.insert(0, str(_INFERENCE_DIR))

from rag_engine import RAGEngine  # noqa: E402
from .schemas import AskRequest, AskResponse  # noqa: E402

engine: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the RAG engine once
    global engine
    print("[main] Starting up -- loading RAG engine (this takes a few seconds)...")
    engine = RAGEngine()
    print("[main] Startup complete.")
    yield
    # Shutdown: nothing to clean up yet
    print("[main] Shutting down.")


app = FastAPI(title="Constitution AI", lifespan=lifespan)

# Add CORS middleware for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "Constitution AI backend is running"}


@app.get("/health")
def health():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {
        "status": "ok",
        "chunks_indexed": engine.retriever.store.index.ntotal,
        "ollama_available": engine.ollama.is_available(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = engine.answer(request.query)
    return result