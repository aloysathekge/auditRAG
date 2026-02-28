"""Cohere reranker for retrieved chunks."""
from auditrag.core.config import get_settings

COHERE_RERANK_MODEL = "rerank-v3.5"


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank chunks using Cohere. Returns top_k with updated scores. Pass-through if no API key."""
    settings = get_settings()
    if not settings.cohere_api_key or not chunks:
        return chunks[:top_k]

    try:
        from cohere import Client
        client = Client(api_key=settings.cohere_api_key)
        documents = [c["text"] for c in chunks]
        response = client.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=min(top_k, len(chunks)),
        )
        reranked = []
        for r in response.results:
            c = chunks[r.index].copy()
            c["score"] = r.relevance_score
            reranked.append(c)
        return reranked
    except Exception:
        return chunks[:top_k]
