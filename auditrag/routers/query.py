import time

from fastapi import APIRouter, HTTPException

from auditrag.generation.llm import generate_answer
from auditrag.observability.telemetry import log_query
from auditrag.retrieval import search
from auditrag.schemas.query import QueryRequest

router = APIRouter(tags=["query"])


@router.post("/query")
def post_query(body: QueryRequest) -> dict:
    """Retrieve chunks and optionally generate an answer with citations."""
    return _run_query(body.question.strip(), body.top_k, body.generate_answer, body.knowledge_base, body.doc_name)


@router.get("/query")
def get_query(
    question: str = "",
    top_k: int = 5,
    generate_answer: bool = True,
    knowledge_base: str | None = None,
) -> dict:
    """Same as POST but with query params."""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="question is required (e.g. ?question=...)")
    top_k = max(1, min(top_k, 20))
    return _run_query(question.strip(), top_k, generate_answer, knowledge_base)


def _run_query(question: str, top_k: int, do_generate: bool, knowledge_base: str | None = None, doc_name: str | None = None) -> dict:
    t0 = time.perf_counter()
    try:
        chunks = search(question, top_k=top_k, knowledge_base=knowledge_base, doc_name=doc_name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e!s}")
    retrieve_ms = round((time.perf_counter() - t0) * 1000)

    out = {"question": question, "chunks": chunks, "latency_ms": {"retrieve_ms": retrieve_ms, "total_ms": retrieve_ms}}
    generate_ms = None
    cost_usd = None
    model_used = None
    if do_generate:
        t1 = time.perf_counter()
        gen = generate_answer(question, chunks)
        generate_ms = round((time.perf_counter() - t1) * 1000)
        out["latency_ms"] = {"retrieve_ms": retrieve_ms, "generate_ms": generate_ms, "total_ms": retrieve_ms + generate_ms}
        if gen:
            out["answer"] = gen["answer"]
            out["sources"] = gen["sources"]
            if gen.get("usage") is not None:
                out["usage"] = gen["usage"]
            if gen.get("cost_usd") is not None:
                out["cost_usd"] = gen["cost_usd"]
            cost_usd = gen.get("cost_usd")
            model_used = gen.get("model_used")
        else:
            cost_usd = None
            model_used = None

    total_ms = out["latency_ms"]["total_ms"]
    log_query(
        question=question,
        total_ms=total_ms,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        cost_usd=cost_usd,
        model_used=model_used,
    )
    return out
