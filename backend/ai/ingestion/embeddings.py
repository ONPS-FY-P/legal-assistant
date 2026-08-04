"""
embeddings.py

Purpose (SDD Section 5 & 8 -- Embedding Workflow):
- Load the chunked Constitution dataset (Articles + Schedules).
- Generate a MiniLM embedding vector for each chunk's text.
- Save embeddings + the chunk metadata together, so vector_store.py
  can build the FAISS index from this output without re-embedding.

This is an OFFLINE step -- run manually whenever the dataset changes
(e.g. when BNS/BNSS/BSA/MVA are added in Sprint 2). It is NOT called
per-request; the API never re-embeds anything at inference time.
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Paths are built relative to THIS FILE's location, not the current working
# directory -- so this script works no matter which folder you run it from
# (backend\, ai\ingestion\, wherever). This avoids the classic "works on my
# machine depending on cwd" bug.
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # ai/ingestion -> ai -> backend
DATASET_PATH = BACKEND_ROOT / "datasets" / "processed" / "constitution_full_dataset.json"
OUTPUT_EMBEDDINGS_PATH = BACKEND_ROOT / "datasets" / "vector_db" / "embeddings.npy"
OUTPUT_METADATA_PATH = BACKEND_ROOT / "datasets" / "vector_db" / "metadata.json"


def load_dataset(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path.resolve()}\n"
            f"Make sure constitution_full_dataset.json is in backend/datasets/processed/"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_embeddings(chunks: list[dict], model_name: str = MODEL_NAME) -> np.ndarray:
    """
    Generate embeddings for all chunks in a single batched call.
    Batching (not looping per-chunk) is what keeps this fast -- MiniLM
    on CPU can comfortably batch hundreds of short legal-text chunks.
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via inner product later
    )
    print(f"Done in {round(time.time() - t0, 1)}s. Shape: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    chunks = load_dataset(DATASET_PATH)
    print(f"Loaded {len(chunks)} chunks from dataset.")

    embeddings = generate_embeddings(chunks)

    Path(OUTPUT_EMBEDDINGS_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_EMBEDDINGS_PATH, embeddings)

    # Save metadata in the SAME order as embeddings, so row i of the
    # embeddings matrix always corresponds to metadata[i]. This ordering
    # contract is critical -- vector_store.py depends on it.
    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings to {OUTPUT_EMBEDDINGS_PATH}")
    print(f"Saved metadata to {OUTPUT_METADATA_PATH}")
    print(f"Embedding dimension: {embeddings.shape[1]}")