"""Hybrid retrieval: dense + sparse with RRF fusion."""
from auditrag.retrieval.dense import search as search_dense
from auditrag.retrieval.sparse import search_sparse

RRF_K = 60


def _chunk_key(c: dict) -> tuple:
    """Unique key for deduplication across dense and sparse results."""
    return (c["doc_name"], c["page"], c["text"][:100])


def _rrf_score(ranks: list[int], k: int = RRF_K) -> float:
    return sum(1.0 / (k + r) for r in ranks)


def search_hybrid(query: str, top_k: int = 5) -> list[dict]:
    """Dense + sparse with Reciprocal Rank Fusion. Returns top_k fused results."""
    fetch_k = max(top_k * 3, 20)
    dense = search_dense(query, top_k=fetch_k)
    sparse = search_sparse(query, top_k=fetch_k)

    rrf_scores: dict[tuple, tuple[dict, list[int]]] = {}

    for rank, c in enumerate(dense):
        key = _chunk_key(c)
        if key not in rrf_scores:
            rrf_scores[key] = (c, [])
        rrf_scores[key][1].append(rank)

    for rank, c in enumerate(sparse):
        key = _chunk_key(c)
        if key not in rrf_scores:
            rrf_scores[key] = (c, [])
        rrf_scores[key][1].append(rank)

    scored = [(chunk, _rrf_score(ranks)) for chunk, ranks in rrf_scores.values()]
    scored.sort(key=lambda x: x[1], reverse=True)

    result = []
    for chunk, score in scored[:top_k]:
        result.append({
            "text": chunk["text"],
            "doc_name": chunk["doc_name"],
            "page": chunk["page"],
            "score": score,
        })
    return result
