from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from auditrag.core.config import get_settings
from auditrag.ingestion.embedder import QDRANT_COLLECTION, QDRANT_TIMEOUT, embed_texts


def search(query: str, top_k: int = 5, knowledge_base: str | None = None) -> list[dict]:
    """Embed the query, search Qdrant, return top_k chunks with text, doc_name, page, score."""
    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
        timeout=QDRANT_TIMEOUT,
    )
    query_vector = embed_texts([query])[0]

    query_filter = None
    if knowledge_base is not None:
        query_filter = Filter(
            must=[FieldCondition(key="knowledge_base", match=MatchValue(value=knowledge_base))]
        )

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return [
        {
            "text": r.payload["text"],
            "doc_name": r.payload["doc_name"],
            "page": r.payload["page"],
            "score": r.score,
        }
        for r in response.points
    ]
