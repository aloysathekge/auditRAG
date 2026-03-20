"""GET /evaluate: run evaluation against auto-generated Q&A pairs."""
from fastapi import APIRouter, Query

from auditrag.evaluation.harness import run_eval, load_qa_pairs

router = APIRouter(tags=["evaluate"])


@router.get("/evaluate")
def get_evaluate(
    doc_name: str | None = Query(None, description="Evaluate only this document"),
    knowledge_base: str | None = Query(None, description="Filter by knowledge base"),
    limit: int = Query(10, ge=1, le=100, description="Max Q&A pairs to evaluate"),
    top_k: int = Query(5, ge=1, le=20, description="Chunks to retrieve per question"),
) -> dict:
    """Run the RAG pipeline against stored Q&A pairs and return metrics."""
    metrics = run_eval(
        doc_name=doc_name,
        knowledge_base=knowledge_base,
        limit=limit,
        top_k=top_k,
        save_json=True,
    )
    return metrics


@router.get("/evaluate/pairs")
def get_qa_pairs(
    doc_name: str | None = Query(None),
    knowledge_base: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    """List stored Q&A pairs (for inspection before running eval)."""
    return load_qa_pairs(doc_name=doc_name, knowledge_base=knowledge_base, limit=limit)
