from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = Path("data/processed/faiss.index")
META_PATH = Path("data/processed/metadata.json")

def build_index(chunks: list[dict]) -> None:
    """Embed all chunks and save FAISS index + metadata to disk."""
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True)
    vectors = np.array(vectors, dtype="float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    metadata = [c["metadata"] | {"text": c["text"]} for c in chunks]
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved index ({index.ntotal} vectors) to {INDEX_PATH}")