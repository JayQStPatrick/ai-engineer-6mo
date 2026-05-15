from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = Path("data/processed/faiss.index")
META_PATH = Path("data/processed/metadata.json")

_model = None
_index = None
_metadata = None

def _load():
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(str(INDEX_PATH))
        _metadata = json.loads(META_PATH.read_text())

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Find the top-k most relevant chunks for a query."""
    _load()
    vec = _model.encode([query])
    vec = np.array(vec, dtype="float32")
    distances, indices = _index.search(vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        item = _metadata[idx]
        results.append({
            "text": item["text"],
            "citation": _format_citation(item),
            "score": float(dist),
            "metadata": item,
        })
    return results

def _format_citation(meta: dict) -> str:
    source = Path(meta["source"]).name
    page = meta.get("page", "")
    section = meta.get("section", "")
    parts = [source]
    if page:
        parts.append(f"p.{page}")
    if section:
        parts.append(f'"{section[:40]}"')
    return " | ".join(parts)