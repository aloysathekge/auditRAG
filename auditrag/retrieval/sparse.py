"""BM25 sparse retrieval over Qdrant corpus."""
import re

from rank_bm25 import BM25Okapi

from auditrag.ingestion.embedder import QDRANT_COLLECTION, QDRANT_TIMEOUT
from auditrag.core.config import get_settings
from qdrant_client import QdrantClient


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


_corpus_cache: list[dict] | None = None
_bm25_index: BM25Okapi | None = None


def _build_index() -> tuple[list[dict], BM25Okapi]:
    """Load all chunks from Qdrant and build BM25 index. Cached."""
    global _corpus_cache, _bm25_index
    if _corpus_cache is not None and _bm25_index is not None:
        return _corpus_cache, _bm25_index

    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
        timeout=QDRANT_TIMEOUT,
    )
    corpus: list[dict] = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
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
        _corpus_cache = []
        _bm25_index = BM25Okapi([[]])
        return _corpus_cache, _bm25_index

    tokenized_corpus = [_tokenize(c["text"]) for c in corpus]
    _corpus_cache = corpus
    _bm25_index = BM25Okapi(tokenized_corpus)
    return _corpus_cache, _bm25_index


def invalidate_sparse_cache() -> None:
    """Call after ingesting new docs so BM25 index is rebuilt on next search."""
    global _corpus_cache, _bm25_index
    _corpus_cache = None
    _bm25_index = None


def search_sparse(query: str, top_k: int = 5) -> list[dict]:
    """BM25 search over Qdrant corpus. Returns same shape as dense.search()."""
    corpus, bm25 = _build_index()
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
