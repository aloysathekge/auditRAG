"""Run evaluation harness: uv run python scripts/run_eval.py [--limit N] (from project root)"""
import argparse

from auditrag.evaluation.harness import run_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Max questions (0 = all)")
    parser.add_argument("--no-save", action="store_true", help="Do not write eval_results/*.json")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    metrics = run_eval(limit=limit, save_json=not args.no_save)

    print("\n--- Evaluation summary ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()


if __name__ == "__main__":
    main()
