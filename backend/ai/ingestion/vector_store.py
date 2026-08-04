"""
vector_store.py

Purpose (SDD Section 5 & 9 -- FAISS Workflow):
- Build a FAISS index from the embeddings produced by embeddings.py.
- Persist the index to disk so the API loads it once at startup
  (never rebuilds it per-request).
- Provide a small wrapper class used by BOTH the offline index-build
  step and the online retriever, so there is one source of truth for
  "how do we talk to FAISS."

Index choice: IndexFlatIP (inner product) on NORMALIZED vectors.
Since embeddings.py already L2-normalizes vectors, inner product IS
cosine similarity -- this is the standard, simple, exact-search choice
for a dataset this size (~700 chunks). If the dataset grows into the
tens of thousands (e.g. after BNS/BNSS/BSA/MVA are added), swap this
for IndexHNSWFlat without changing any other module -- that's the
whole point of isolating FAISS logic here.
"""

import json
from pathlib import Path

import faiss
import numpy as np

# Paths are built relative to THIS FILE's location, not the current working
# directory -- consistent with embeddings.py, so both scripts work no matter
# which folder you run them from.
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # ai/ingestion -> ai -> backend
EMBEDDINGS_PATH = BACKEND_ROOT / "datasets" / "vector_db" / "embeddings.npy"
METADATA_PATH = BACKEND_ROOT / "datasets" / "vector_db" / "metadata.json"
INDEX_PATH = BACKEND_ROOT / "datasets" / "vector_db" / "index.faiss"


class VectorStore:
    def __init__(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Build a fresh FAISS index from embeddings + matching metadata."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Embeddings/metadata count mismatch: "
                f"{embeddings.shape[0]} vectors vs {len(metadata)} metadata entries"
            )

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        self.index = index
        self.metadata = metadata

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("No index to save -- call build() first")
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """Load a previously built index + metadata from disk."""
        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"No index found at {self.index_path} -- run build_index.py first")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search the index for the top_k most similar chunks.
        query_vector must already be normalized the same way as the
        indexed embeddings (see embeddings.py's normalize_embeddings=True).
        """
        if self.index is None:
            raise RuntimeError("Index not loaded -- call load() or build() first")

        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)
        return results


if __name__ == "__main__":
    # Manual build step: load embeddings.py's output, build the index, save it.
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Building FAISS index from {embeddings.shape[0]} vectors (dim={embeddings.shape[1]})")

    store = VectorStore()
    store.build(embeddings, metadata)
    store.save()

    print(f"Index saved to {INDEX_PATH}")

    # Sanity check: search using the first chunk's own embedding --
    # it should return itself as the top result with score ~1.0
    test_vector = embeddings[0]
    results = store.search(test_vector, top_k=3)
    print("\nSanity check -- searching with chunk 0's own vector:")
    for r in results:
        print(f"  score={r['similarity_score']:.4f}  {r.get('title', '')[:60]}")