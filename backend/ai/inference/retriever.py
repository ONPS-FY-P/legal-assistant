"""
retriever.py

Purpose (SDD Section 5 -- RAG Pipeline, retrieval stage):
- Take a user's natural-language legal question.
- Embed it with the SAME model used to build the index (MiniLM) --
  using a different model here would make the vectors incomparable.
- Search the FAISS index (via VectorStore) for the top-k most relevant
  Article/Schedule chunks.

This is an ONLINE module -- it runs on every single /ask request, so
it loads the model and index ONCE (at import/startup) and reuses them,
rather than reloading per request.
"""

"""
retriever.py

Purpose (SDD Section 5 -- RAG Pipeline, retrieval stage):
- Take a user's natural-language legal question.
- Run TWO parallel search methods and fuse their results:
    1. DENSE search (FAISS + MiniLM) -- understands meaning/paraphrasing,
       but is sensitive to exact phrasing (short/casual queries can score
       lower than expected -- this is what we saw with "What is my right
       to life?" scoring 0.418, below a pure-dense threshold).
    2. BM25 keyword search -- exact term matching, robust to short
       queries, catches things dense search under-scores (e.g. "life"
       literally appearing in Article 21's text).
- Combine both rankings with Reciprocal Rank Fusion (RRF), which doesn't
  require normalizing two differently-scaled scores (cosine similarity
  vs BM25 score) -- it just combines RANK POSITIONS, which is simpler
  and more robust than trying to weight raw scores against each other.

This is an ONLINE module -- loads the model, FAISS index, and BM25
index ONCE at startup, reused for every request.
"""

import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_INGESTION_DIR = Path(__file__).resolve().parent.parent / "ingestion"
sys.path.insert(0, str(_INGESTION_DIR))

from vector_store import VectorStore  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
DENSE_CANDIDATES = 20  # how many dense results to pull before fusion
BM25_CANDIDATES = 20   # how many BM25 results to pull before fusion
RRF_K = 60              # standard RRF constant -- dampens the impact of any single rank

STOPWORDS = set("""a an the is are was were be been being of to in on at for with by from as
what which who whom this that these those my your his her its our their i you he she it we they
do does did have has had will would shall should can could may might must not no and or but if
""".split())


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


class Retriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

        print(f"[retriever] Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

        print("[retriever] Loading FAISS index...")
        self.store = VectorStore()
        self.store.load()
        print(f"[retriever] FAISS ready. {self.store.index.ntotal} chunks.")

        print("[retriever] Building BM25 keyword index...")
        corpus_tokens = [
            _tokenize(c["text"] + " " + (c.get("title") or ""))
            for c in self.store.metadata
        ]
        self.bm25 = BM25Okapi(corpus_tokens)
        print("[retriever] BM25 ready.")

    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Returns [(chunk_index, cosine_score), ...] sorted by score desc."""
        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].reshape(1, -1)
        scores, indices = self.store.index.search(query_vector.astype("float32"), k)
        return [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Returns [(chunk_index, bm25_score), ...] sorted by score desc."""
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in top_idx]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Hybrid retrieval: fuse dense + BM25 rankings via RRF, return the
        top_k chunks. Each result carries BOTH raw scores (when available)
        plus the fused rrf_score used for final ranking -- rag_engine.py's
        confidence gate uses the raw scores, not rrf_score, since rrf_score
        isn't on an interpretable 0-1 scale.
        """
        k = top_k or self.top_k

        dense_results = self._dense_search(query, DENSE_CANDIDATES)
        bm25_results = self._bm25_search(query, BM25_CANDIDATES)

        dense_scores = {idx: score for idx, score in dense_results}
        bm25_scores = {idx: score for idx, score in bm25_results}

        # RRF: for each method, a chunk's contribution is 1/(RRF_K + rank),
        # rank starting at 1 for the top result in that method's own list.
        rrf_scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (idx, _) in enumerate(bm25_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        ranked_indices = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])[:k]

        results = []
        for idx in ranked_indices:
            chunk = self.store.metadata[idx].copy()
            chunk["dense_score"] = dense_scores.get(idx)   # None if not in dense top-N
            chunk["bm25_score"] = bm25_scores.get(idx)      # None if not in BM25 top-N
            chunk["rrf_score"] = rrf_scores[idx]
            results.append(chunk)
        return results


if __name__ == "__main__":
    retriever = Retriever(top_k=3)

    test_queries = [
        "What is my right to life?",                              # previously failed pure-dense (0.418)
        "What are the fundamental rights of Indian citizens?",
        "What languages are officially recognized in India?",
        "What is the punishment for not wearing a helmet on a bike?",  # should still show weak signal both ways
    ]

    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        results = retriever.retrieve(q)
        for r in results:
            label = f"Article {r['article_number']}" if r.get("article_number") else r.get("schedule_name", "")
            d = f"{r['dense_score']:.3f}" if r["dense_score"] is not None else "  -  "
            b = f"{r['bm25_score']:.2f}" if r["bm25_score"] is not None else "  -  "
            print(f"  rrf={r['rrf_score']:.4f}  dense={d}  bm25={b}  {label} -- {r.get('title','')[:50]}")