"""Interactive CLI: uv run python -m auditrag"""
import sys
import time

from auditrag.generation.llm import generate_answer
from auditrag.retrieval.dense import search

TOP_K = 5


def run_one(question: str, generate: bool = True) -> None:
    question = question.strip()
    if not question:
        return
    t0 = time.perf_counter()
    print("Searching...")
    results = search(question, top_k=TOP_K)
    retrieve_ms = round((time.perf_counter() - t0) * 1000)
    print(f"Found {len(results)} chunks ({retrieve_ms} ms)\n")

    if generate:
        t1 = time.perf_counter()
        gen = generate_answer(question, results)
        generate_ms = round((time.perf_counter() - t1) * 1000)
        if gen:
            print("--- Answer ---")
            print(gen["answer"])
            print("\n--- Sources ---")
            for s in gen["sources"]:
                print(f"  {s['doc_name']} p{s['page']}")
            line = f"\nLatency: retrieve={retrieve_ms} ms, generate={generate_ms} ms, total={retrieve_ms + generate_ms} ms"
            if gen.get("usage"):
                line += f" | Tokens: in={gen['usage']['input_tokens']} out={gen['usage']['output_tokens']}"
            if gen.get("cost_usd") is not None:
                line += f" | Cost: ${gen['cost_usd']:.6f}"
            print(line + "\n")
        else:
            print("(Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env to generate an answer.)\n")

    for i, r in enumerate(results, 1):
        preview = r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"]
        print(f"--- Chunk {i} (score={r['score']:.4f}, {r['doc_name']} p{r['page']}) ---")
        print(preview)
        print()


def main() -> None:
    print("auditRAG — type a question (blank line or Ctrl+C to exit)\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            sys.exit(0)
        if not question:
            continue
        run_one(question, generate=True)


if __name__ == "__main__":
    main()
