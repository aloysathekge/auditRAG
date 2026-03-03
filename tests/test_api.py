from unittest.mock import patch

from fastapi.testclient import TestClient

from auditrag.main import app


client = TestClient(app)


def test_health_endpoint_ok_shape():
    with (
        patch("auditrag.routers.health.check_postgres", return_value=(True, None)),
        patch("auditrag.routers.health.check_qdrant", return_value=(True, None)),
    ):
        resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "checks" in data
    assert "postgres" in data["checks"]
    assert "qdrant" in data["checks"]


def test_get_query_requires_question_param():
    resp = client.get("/query")
    assert resp.status_code == 400
    assert "question is required" in resp.json()["detail"]


def test_post_query_calls_retrieval_and_generation():
    fake_chunks = [{"doc_name": "doc", "page": 1, "text": "some text", "score": 0.9}]
    fake_answer = {
        "answer": "Here is the answer.",
        "sources": [{"doc_name": "doc", "page": 1}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "cost_usd": 0.001,
        "model_used": "gpt-4o-mini",
    }

    with (
        patch("auditrag.routers.query.search", return_value=fake_chunks),
        patch("auditrag.routers.query.generate_answer", return_value=fake_answer),
        patch("auditrag.routers.query.log_query") as mock_log_query,
    ):
        resp = client.post("/query", json={"question": "What is revenue?", "top_k": 3, "generate_answer": True})

    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == "What is revenue?"
    assert data["chunks"] == fake_chunks
    assert data["answer"] == fake_answer["answer"]
    assert data["sources"] == fake_answer["sources"]
    assert data["usage"] == fake_answer["usage"]
    assert data["cost_usd"] == fake_answer["cost_usd"]
    assert "latency_ms" in data
    assert "retrieve_ms" in data["latency_ms"]
    assert "total_ms" in data["latency_ms"]

    mock_log_query.assert_called_once()

