from unittest.mock import MagicMock, patch

from auditrag.retrieval.search import search


class DummySettings:
    def __init__(self, retrieval_mode: str = "hybrid", use_reranker: bool = False):
        self.retrieval_mode = retrieval_mode
        self.use_reranker = use_reranker


def _make_chunk(label: str) -> dict:
    return {"text": f"text-{label}", "doc_name": f"doc-{label}", "page": 1, "score": 1.0}


def test_search_uses_dense_when_mode_dense():
    dense_result = [_make_chunk("dense")]

    with (
        patch("auditrag.retrieval.search.get_settings", return_value=DummySettings("dense", False)),
        patch("auditrag.retrieval.search.search_dense", return_value=dense_result) as mock_dense,
        patch("auditrag.retrieval.search.search_sparse") as mock_sparse,
        patch("auditrag.retrieval.search.search_hybrid") as mock_hybrid,
        patch("auditrag.retrieval.search.rerank_chunks") as mock_rerank,
    ):
        out = search("what is revenue?", top_k=5)

    assert out == dense_result
    mock_dense.assert_called_once_with("what is revenue?", top_k=5, knowledge_base=None)
    mock_sparse.assert_not_called()
    mock_hybrid.assert_not_called()
    mock_rerank.assert_not_called()


def test_search_uses_sparse_when_mode_sparse():
    sparse_result = [_make_chunk("sparse")]

    with (
        patch("auditrag.retrieval.search.get_settings", return_value=DummySettings("sparse", False)),
        patch("auditrag.retrieval.search.search_dense") as mock_dense,
        patch("auditrag.retrieval.search.search_sparse", return_value=sparse_result) as mock_sparse,
        patch("auditrag.retrieval.search.search_hybrid") as mock_hybrid,
        patch("auditrag.retrieval.search.rerank_chunks") as mock_rerank,
    ):
        out = search("what is revenue?", top_k=5)

    assert out == sparse_result
    mock_sparse.assert_called_once_with("what is revenue?", top_k=5, knowledge_base=None)
    mock_dense.assert_not_called()
    mock_hybrid.assert_not_called()
    mock_rerank.assert_not_called()


def test_search_uses_hybrid_by_default_and_when_mode_hybrid():
    hybrid_result = [_make_chunk("hybrid")]

    with (
        patch("auditrag.retrieval.search.get_settings", return_value=DummySettings("hybrid", False)),
        patch("auditrag.retrieval.search.search_dense") as mock_dense,
        patch("auditrag.retrieval.search.search_sparse") as mock_sparse,
        patch("auditrag.retrieval.search.search_hybrid", return_value=hybrid_result) as mock_hybrid,
        patch("auditrag.retrieval.search.rerank_chunks") as mock_rerank,
    ):
        out1 = search("question", top_k=3)

    assert out1 == hybrid_result
    mock_hybrid.assert_called_once_with("question", top_k=3, knowledge_base=None)
    mock_dense.assert_not_called()
    mock_sparse.assert_not_called()
    mock_rerank.assert_not_called()

    # Unrecognised mode -> falls back to hybrid
    with (
        patch("auditrag.retrieval.search.get_settings", return_value=DummySettings("weird-mode", False)),
        patch("auditrag.retrieval.search.search_dense") as mock_dense2,
        patch("auditrag.retrieval.search.search_sparse") as mock_sparse2,
        patch("auditrag.retrieval.search.search_hybrid", return_value=hybrid_result) as mock_hybrid2,
        patch("auditrag.retrieval.search.rerank_chunks") as mock_rerank2,
    ):
        out2 = search("question", top_k=3)

    assert out2 == hybrid_result
    mock_hybrid2.assert_called_once()
    mock_dense2.assert_not_called()
    mock_sparse2.assert_not_called()
    mock_rerank2.assert_not_called()


def test_search_calls_reranker_when_enabled_and_chunks_present():
    base_chunks = [_make_chunk("base")]
    reranked = [_make_chunk("reranked")]

    with (
        patch("auditrag.retrieval.search.get_settings", return_value=DummySettings("dense", True)),
        patch("auditrag.retrieval.search.search_dense", return_value=base_chunks),
        patch("auditrag.retrieval.search.rerank_chunks", return_value=reranked) as mock_rerank,
    ):
        out = search("some question", top_k=5)

    assert out == reranked
    mock_rerank.assert_called_once()


def test_search_passes_knowledge_base_to_dense():
    dense_result = [_make_chunk("dense")]

    with (
        patch("auditrag.retrieval.search.get_settings", return_value=DummySettings("dense", False)),
        patch("auditrag.retrieval.search.search_dense", return_value=dense_result) as mock_dense,
        patch("auditrag.retrieval.search.search_sparse") as mock_sparse,
        patch("auditrag.retrieval.search.search_hybrid") as mock_hybrid,
        patch("auditrag.retrieval.search.rerank_chunks") as mock_rerank,
    ):
        out = search("question", top_k=5, knowledge_base="my_kb")

    mock_dense.assert_called_once_with("question", top_k=5, knowledge_base="my_kb")
