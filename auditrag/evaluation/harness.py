"""Run evaluation against auto-generated Q&A pairs from ingested documents."""
import json
import time
from datetime import datetime
from pathlib import Path

from auditrag.db.models import QAPair
from auditrag.db.session import SessionLocal
from auditrag.generation.llm import generate_answer
from auditrag.retrieval import search

RESULTS_DIR = Path("eval_results")


def run_one(question: str, top_k: int = 5, knowledge_base: str | None = None) -> dict:
    """Run retrieve + generate for one question. Returns result dict."""
    t0 = time.perf_counter()
    chunks = search(question, top_k=top_k, knowledge_base=knowledge_base)
    retrieve_ms = round((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    gen = generate_answer(question, chunks, mode="concise")
    generate_ms = round((time.perf_counter() - t1) * 1000) if gen else 0

    answer = (gen.get("answer") or "").strip() if gen else ""
    cost_usd = gen.get("cost_usd") if gen else None

    return {
        "question": question,
        "answer": answer,
        "num_chunks": len(chunks),
        "retrieve_ms": retrieve_ms,
        "generate_ms": generate_ms,
        "total_ms": retrieve_ms + generate_ms,
        "cost_usd": cost_usd,
    }


def load_qa_pairs(
    doc_name: str | None = None,
    knowledge_base: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Load auto-generated Q&A pairs from the database."""
    session = SessionLocal()
    try:
        query = session.query(QAPair)
        if doc_name:
            query = query.filter(QAPair.doc_name == doc_name)
        if knowledge_base:
            query = query.filter(QAPair.knowledge_base == knowledge_base)
        if limit:
            query = query.limit(limit)
        rows = query.all()
        return [
            {
                "question": r.question,
                "gold_answer": r.answer,
                "doc_name": r.doc_name,
                "knowledge_base": r.knowledge_base,
                "chunk_id": r.chunk_id,
            }
            for r in rows
        ]
    finally:
        session.close()


def run_harness(
    doc_name: str | None = None,
    knowledge_base: str | None = None,
    limit: int | None = 10,
    top_k: int = 5,
) -> list[dict]:
    """Load Q&A pairs from DB, run each through search + generate, return results."""
    qa_pairs = load_qa_pairs(doc_name=doc_name, knowledge_base=knowledge_base, limit=limit)
    if not qa_pairs:
        return []

    results = []
    for pair in qa_pairs:
        out = run_one(pair["question"], top_k=top_k, knowledge_base=knowledge_base)
        out["gold_answer"] = pair["gold_answer"]
        out["source_doc"] = pair["doc_name"]
        results.append(out)

    return results


def run_eval(
    doc_name: str | None = None,
    knowledge_base: str | None = None,
    limit: int | None = 10,
    top_k: int = 5,
    save_json: bool = True,
) -> dict:
    """Run harness, compute metrics, optionally save results to JSON, return summary."""
    from auditrag.evaluation.metrics import compute_metrics

    results = run_harness(doc_name=doc_name, knowledge_base=knowledge_base, limit=limit, top_k=top_k)
    metrics = compute_metrics(results)

    if save_json and results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"eval_{ts}.json"
        with open(path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        metrics["results_path"] = str(path)

    metrics["details"] = results
    return metrics
