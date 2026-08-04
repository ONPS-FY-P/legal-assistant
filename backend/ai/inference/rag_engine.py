"""
rag_engine.py

Purpose (SDD Section 5 -- RAG Pipeline orchestration):
- The single entry point that ties retrieval, prompt-building, and
  Ollama generation together.
- This is what api/routes/ask.py will call later -- the API layer
  should know nothing about FAISS, embeddings, or prompts; it just
  calls rag_engine.answer(query) and gets a result back.

Confidence gate (hybrid-aware):
- retriever.py now returns BOTH a dense (cosine) score and a BM25
  score per chunk, since a query can be genuinely relevant via ONE
  method even when weak in the other (e.g. "What is my right to life?"
  scored only 0.418 on pure dense search, but scores strongly on BM25
  because "life" and "right" appear directly in Article 21's text).
- We only refuse to answer (trigger the gate) if the top result is
  weak on BOTH signals -- that's a much more reliable "this genuinely
  isn't in the dataset" signal than either method alone.

Web Search Integration:
- After generating the Constitution-based answer from Llama (100% trusted),
  we fetch practical suggestions from the web (FIR filing, complaint procedures,
  government portals, etc.) using DuckDuckGo Search.
- This keeps the verified constitutional content separate from practical guidance.
"""

from retriever import Retriever
from prompt_builder import build_prompt
from ollama_client import OllamaClient
from web_search import WebSearchClient, build_practical_suggestions_query

MIN_DENSE_SIMILARITY = 0.40   # cosine similarity floor
MIN_BM25_SCORE = 6.0           # empirically: real matches scored 9-12 in our testing,
                                 # unrelated topics (e.g. Motor Vehicles Act questions) scored near 0
TOP_K = 5                       # how many candidates retriever.py fetches
MAX_CONTEXT_CHUNKS = 3          # how many of those actually go into the LLM prompt --
                                  # capped lower than TOP_K because testing showed the model
                                  # (Llama 3.2) is more likely to pick the wrong source to cite
                                  # when given too many plausible-looking candidates at once

INSUFFICIENT_INFO_MESSAGE = (
    "I don't have enough information in the Constitution to answer this confidently. "
    "This may be covered under a different law not yet in my database "
    "(e.g. Motor Vehicles Act, BNS, BNSS, BSA -- coming in a future update)."
)


class RAGEngine:
    def __init__(self):
        print("[rag_engine] Initializing...")
        self.retriever = Retriever(top_k=TOP_K)
        self.ollama = OllamaClient()
        self.web_search = WebSearchClient()

        if not self.ollama.is_available():
            print("[rag_engine] WARNING: Ollama is not reachable. "
                  "Retrieval will work but generation will fail until Ollama is running.")
        
        if not self.web_search.is_available():
            print("[rag_engine] WARNING: DuckDuckGo web search is not available. "
                  "Practical suggestions will be unavailable until internet access is restored.")
        else:
            print("[rag_engine] Web search ready.")
        
        print("[rag_engine] Ready.")

    def _is_confident(self, top_chunk: dict) -> bool:
        """Confident if EITHER signal clears its bar -- see module docstring."""
        dense_ok = top_chunk["dense_score"] is not None and top_chunk["dense_score"] >= MIN_DENSE_SIMILARITY
        bm25_ok = top_chunk["bm25_score"] is not None and top_chunk["bm25_score"] >= MIN_BM25_SCORE
        return dense_ok or bm25_ok

    def answer(self, query: str) -> dict:
        """
        Full RAG pipeline: retrieve -> gate on confidence -> build prompt -> generate.
        Then fetch practical web suggestions (FIR filing, complaint procedures, etc.).
        
        Response structure:
        - constitution_answer: The Llama-generated answer (100% trusted, from Constitution)
        - practical_suggestions: Web-search results with actionable steps
        - sources_used: Constitutional articles/schedules cited
        """
        chunks = self.retriever.retrieve(query)

        if not chunks or not self._is_confident(chunks[0]):
            return {
                "query": query,
                "constitution_answer": INSUFFICIENT_INFO_MESSAGE,
                "practical_suggestions": [],
                "sources_used": [],
                "confidence_gate_triggered": True,
                "top_dense_score": chunks[0]["dense_score"] if chunks else None,
                "top_bm25_score": chunks[0]["bm25_score"] if chunks else None,
            }

        # Keep chunks that individually clear at least one signal, same logic
        # as the gate itself -- avoids passing pure noise into the prompt
        # even when the top result is strong. Then cap to MAX_CONTEXT_CHUNKS,
        # keeping the strongest (list is already best-to-worst from retriever.py).
        relevant_chunks = [c for c in chunks if self._is_confident(c)]
        if not relevant_chunks:
            relevant_chunks = chunks[:1]  # always keep at least the top result
        relevant_chunks = relevant_chunks[:MAX_CONTEXT_CHUNKS]

        prompt = build_prompt(query, relevant_chunks)
        constitution_answer = self.ollama.generate(prompt)
        
        # Extract topic from chunks for practical suggestions query
        primary_source = relevant_chunks[0]
        if primary_source.get("article_number"):
            constitutional_topic = f"Article {primary_source['article_number']} {primary_source.get('title', '')}"
        elif primary_source.get("schedule_name"):
            constitutional_topic = f"{primary_source['schedule_name']} {primary_source.get('title', '')}"
        else:
            constitutional_topic = query
        
        # Fetch practical suggestions from web
        practical_query = build_practical_suggestions_query(constitutional_topic)
        practical_suggestions = self.web_search.search(practical_query)

        return {
            "query": query,
            "constitution_answer": constitution_answer,
            "practical_suggestions": practical_suggestions,
            "sources_used": [
                {
                    "article_number": c.get("article_number"),
                    "schedule_name": c.get("schedule_name"),
                    "title": c.get("title"),
                    "dense_score": c["dense_score"],
                    "bm25_score": c["bm25_score"],
                }
                for c in relevant_chunks
            ],
            "confidence_gate_triggered": False,
            "top_dense_score": chunks[0]["dense_score"],
            "top_bm25_score": chunks[0]["bm25_score"],
        }


if __name__ == "__main__":
    engine = RAGEngine()

    test_queries = [
        "What is my right to life?",  # previously blocked by pure-dense gate
        "What languages are officially recognized in India?",
        "What is the punishment for not wearing a helmet on a bike?",  # should still trigger the gate
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        print('='*70)
        result = engine.answer(q)
        print(f"\nGate triggered: {result['confidence_gate_triggered']} "
              f"(dense: {result['top_dense_score']}, bm25: {result['top_bm25_score']})")
        print(f"\nCONSTITUTION ANSWER (from Llama, 100% trusted):\n{result['constitution_answer']}")
        if result["sources_used"]:
            print(f"\nSources: {[s.get('article_number') or s.get('schedule_name') for s in result['sources_used']]}")
        
        if result["practical_suggestions"]:
            print(f"\nPRACTICAL SUGGESTIONS (from web search):")
            for i, suggestion in enumerate(result["practical_suggestions"], 1):
                print(f"\n  {i}. {suggestion['title']}")
                print(f"     Source: {suggestion['source']}")
                print(f"     URL: {suggestion['url']}")
                print(f"     Info: {suggestion['snippet'][:150]}...")
        else:
            print("\nNo practical suggestions available (web search may be offline).")