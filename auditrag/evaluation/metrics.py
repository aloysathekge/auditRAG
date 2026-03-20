"""Compute evaluation metrics from harness results."""
import re


def clean_answer(text: str) -> str:
    """Strip formatting artifacts before scoring."""
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove citations like [doc_name, page N]
    text = re.sub(r'\[.+?\]', '', text)
    # Remove trailing periods and whitespace
    text = text.strip().rstrip('.')
    return text


def _normalize(s: str) -> str:
    return " ".join(clean_answer(s or "").lower().split())


def token_f1(pred: str, gold: str) -> float:
    """F1 of token overlap (pred vs gold). 0 if either empty."""
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def exact_match(pred: str, gold: str) -> bool:
    """Normalized exact match."""
    return _normalize(pred) == _normalize(gold)


def compute_metrics(results: list[dict]) -> dict:
    """
    Aggregate metrics from harness results.
    results: list of dicts with keys answer, gold_answer, total_ms, cost_usd, etc.
    """
    if not results:
        return {
            "count": 0,
            "exact_match_rate": None,
            "token_f1_avg": None,
            "avg_latency_ms": None,
            "avg_cost_usd": None,
        }

    n = len(results)
    em_count = sum(1 for r in results if exact_match(r.get("answer") or "", r.get("gold_answer") or ""))
    f1_sum = sum(token_f1(r.get("answer") or "", r.get("gold_answer") or "") for r in results)
    total_ms_list = [r["total_ms"] for r in results]
    cost_list = [r["cost_usd"] for r in results if r.get("cost_usd") is not None]

    avg_latency = sum(total_ms_list) / n
    avg_cost = sum(cost_list) / len(cost_list) if cost_list else None

    return {
        "count": n,
        "exact_match_rate": round(em_count / n, 4),
        "token_f1_avg": round(f1_sum / n, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_cost_usd": round(avg_cost, 6) if avg_cost is not None else None,
    }
