"""Unified search: dispatches to dense/sparse/hybrid per config, optional rerank."""
from auditrag.core.config import get_settings
from auditrag.retrieval.dense import search as search_dense
from auditrag.retrieval.sparse import search_sparse
from auditrag.retrieval.hybrid import search_hybrid
from auditrag.retrieval.reranker import rerank as rerank_chunks


def search(query: str, top_k: int = 5, knowledge_base: str | None = None) -> list[dict]:
    """Retrieve top_k chunks. Uses retrieval_mode (dense|sparse|hybrid) and optional reranker."""
    settings = get_settings()
    mode = (settings.retrieval_mode or "hybrid").lower()

    if mode == "dense":
        chunks = search_dense(query, top_k=top_k, knowledge_base=knowledge_base)
    elif mode == "sparse":
        chunks = search_sparse(query, top_k=top_k, knowledge_base=knowledge_base)
    elif mode == "hybrid":
        chunks = search_hybrid(query, top_k=top_k, knowledge_base=knowledge_base)
    else:
        chunks = search_hybrid(query, top_k=top_k, knowledge_base=knowledge_base)

    if settings.use_reranker and chunks:
        chunks = rerank_chunks(query, chunks, top_k=top_k)

    return chunks
