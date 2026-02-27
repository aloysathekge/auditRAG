"""Query from CLI: python -m auditrag.retrieval [question] [-a] [-k N]"""
import argparse
import sys
import time

from auditrag.generation.llm import generate_answer
from auditrag.retrieval.dense import search

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query auditRAG: retrieve chunks and optionally generate an answer.")
    parser.add_argument("question", nargs="*", help="Question (or leave empty to type when prompted)")
    parser.add_argument("-a", "--answer", action="store_true", help="Generate LLM answer from chunks")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    args = parser.parse_args()

    question = " ".join(args.question).strip() if args.question else input("Question: ").strip()
    if not question:
        print("No question given.")
        sys.exit(1)

    t0 = time.perf_counter()
    print("Searching...")
    results = search(question, top_k=args.top_k)
    retrieve_ms = round((time.perf_counter() - t0) * 1000)
    print(f"Found {len(results)} chunks ({retrieve_ms} ms)\n")

    if args.answer:
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
