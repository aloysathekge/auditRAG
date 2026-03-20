"""BM25 sparse retrieval over Qdrant corpus."""
import re

from rank_bm25 import BM25Okapi

from auditrag.ingestion.embedder import QDRANT_COLLECTION, QDRANT_TIMEOUT
from auditrag.core.config import get_settings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# Cache keyed by knowledge_base (None = all)
_cache: dict[str | None, tuple[list[dict], BM25Okapi]] = {}


def _build_index(knowledge_base: str | None = None) -> tuple[list[dict], BM25Okapi]:
    """Load chunks from Qdrant and build BM25 index. Cached per knowledge_base."""
    global _cache
    if knowledge_base in _cache:
        return _cache[knowledge_base]

    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
        timeout=QDRANT_TIMEOUT,
    )

    scroll_filter = None
    if knowledge_base is not None:
        scroll_filter = Filter(
            must=[FieldCondition(key="knowledge_base", match=MatchValue(value=knowledge_base))]
        )

    corpus: list[dict] = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in result:
            payload = pt.payload or {}
            corpus.append({
                "chunk_id": payload.get("chunk_id", ""),
                "doc_name": payload.get("doc_name", ""),
                "page": payload.get("page", 0),
                "text": payload.get("text", ""),
            })
        if next_offset is None:
            break
        offset = next_offset

    if not corpus:
        bm25 = BM25Okapi([[]])
        _cache[knowledge_base] = ([], bm25)
        return [], bm25

    tokenized_corpus = [_tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    _cache[knowledge_base] = (corpus, bm25)
    return corpus, bm25


def invalidate_sparse_cache() -> None:
    """Call after ingesting new docs so BM25 index is rebuilt on next search."""
    global _cache
    _cache.clear()


def search_sparse(query: str, top_k: int = 5, knowledge_base: str | None = None) -> list[dict]:
    """BM25 search over Qdrant corpus. Returns same shape as dense.search()."""
    corpus, bm25 = _build_index(knowledge_base)
    if not corpus:
        return []

    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)
    indices = scores.argsort()[::-1][:top_k]

    return [
        {
            "text": corpus[i]["text"],
            "doc_name": corpus[i]["doc_name"],
            "page": corpus[i]["page"],
            "score": float(scores[indices[i]]),
        }
        for i in range(len(indices))
        if scores[indices[i]] > 0
    ]
