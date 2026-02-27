import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from auditrag.generate import generate_answer
from auditrag.retrieve import search

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to return")
    generate_answer: bool = Field(True, description="If true, return LLM answer + sources (requires OPENAI_API_KEY)")


@router.post("/query")
def post_query(body: QueryRequest) -> dict:
    """Retrieve chunks and optionally generate an answer with citations."""
    return _run_query(body.question.strip(), body.top_k, body.generate_answer)


@router.get("/query")
def get_query(question: str = "", top_k: int = 5, generate_answer: bool = True) -> dict:
    """Same as POST but with query params. Example: GET /query?question=What+was+3M+capital+expenditure"""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="question is required (e.g. ?question=...)")
    top_k = max(1, min(top_k, 20))
    return _run_query(question.strip(), top_k, generate_answer)


def _run_query(question: str, top_k: int, do_generate: bool) -> dict:
    t0 = time.perf_counter()
    try:
        chunks = search(question, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e!s}")
    retrieve_ms = round((time.perf_counter() - t0) * 1000)

    out = {"question": question, "chunks": chunks, "latency_ms": {"retrieve_ms": retrieve_ms, "total_ms": retrieve_ms}}
    if do_generate:
        t1 = time.perf_counter()
        gen = generate_answer(question, chunks)
        generate_ms = round((time.perf_counter() - t1) * 1000)
        out["latency_ms"] = {"retrieve_ms": retrieve_ms, "generate_ms": generate_ms, "total_ms": retrieve_ms + generate_ms}
        if gen:
            out["answer"] = gen["answer"]
            out["sources"] = gen["sources"]
    return out
