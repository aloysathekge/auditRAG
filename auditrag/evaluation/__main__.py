"""Run evaluation: uv run python -m auditrag.evaluation [--limit N] [--no-save]"""
import argparse

from auditrag.evaluation.harness import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinanceBench evaluation harness.")
    parser.add_argument("--limit", type=int, default=10, help="Max number of questions (default 10, use 0 for all)")
    parser.add_argument("--no-save", action="store_true", help="Do not save results JSON")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    metrics = run_eval(limit=limit, save_json=not args.no_save)

    print("\n--- Evaluation summary ---")
    print(f"  count:            {metrics['count']}")
    print(f"  exact_match_rate: {metrics.get('exact_match_rate')}")
    print(f"  token_f1_avg:     {metrics.get('token_f1_avg')}")
    print(f"  avg_latency_ms:   {metrics.get('avg_latency_ms')}")
    print(f"  avg_cost_usd:     {metrics.get('avg_cost_usd')}")
    if metrics.get("results_path"):
        print(f"  results_path:     {metrics['results_path']}")
    print()


if __name__ == "__main__":
    main()
