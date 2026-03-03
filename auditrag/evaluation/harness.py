"""Run FinanceBench Q&A through the full pipeline and collect results."""
import json
import time
from datetime import datetime
from pathlib import Path

from auditrag.generation.llm import generate_answer
from auditrag.retrieval import search

RESULTS_DIR = Path("eval_results")


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace for comparison."""
    return " ".join((s or "").lower().split())


def run_one(question: str, top_k: int = 5) -> dict:
    """Run retrieve + generate for one question. Returns result dict."""
    t0 = time.perf_counter()
    chunks = search(question, top_k=top_k)
    retrieve_ms = round((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    gen = generate_answer(question, chunks)
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


def run_harness(
    limit: int | None = 10,
    top_k: int = 5,
    dataset_split: str = "train",
) -> list[dict]:
    """
    Load FinanceBench, run each question through search + generate, return list of result dicts.
    limit: max number of examples (None = all). Use small limit for quick runs.
    """
    from datasets import load_dataset

    ds = load_dataset("PatronusAI/financebench", split=dataset_split)
    n = len(ds) if limit is None else min(limit, len(ds))
    results = []

    for i in range(n):
        row = ds[i]
        question = (row.get("question") or "").strip()
        if not question:
            continue
        gold_answer = (row.get("answer") or "").strip()
        out = run_one(question, top_k=top_k)
        out["gold_answer"] = gold_answer
        results.append(out)

    return results


def run_eval(
    limit: int | None = 10,
    top_k: int = 5,
    save_json: bool = True,
) -> dict:
    """
    Run harness, compute metrics, optionally save results to JSON, return summary.
    """
    from auditrag.evaluation.metrics import compute_metrics

    results = run_harness(limit=limit, top_k=top_k)
    metrics = compute_metrics(results)

    if save_json and results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"eval_{ts}.json"
        with open(path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        metrics["results_path"] = str(path)

    return metrics
