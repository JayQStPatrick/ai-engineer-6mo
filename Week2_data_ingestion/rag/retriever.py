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