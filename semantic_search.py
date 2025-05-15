import numpy as np
from typing import List


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Beräknar cosinuslikheten mellan två vektorer."""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    denom = np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
    if denom == 0:
        return 0.0
    return float(np.dot(vec1_np, vec2_np) / denom)


def semantic_search(
    query_embedding: List[float],
    chunks: List[str],
    embeddings: List[List[float]],
    top_k: int = 5,
) -> List[str]:
    """Hittar top-k mest relevanta chunk-texter enligt cosinuslikhet."""
    similarities = [
        (idx, cosine_similarity(query_embedding, emb)) for idx, emb in enumerate(embeddings)
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:top_k]]
    return [chunks[i] for i in top_indices]
