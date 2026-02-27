from qdrant_client import QdrantClient

from auditrag.config import get_settings
from auditrag.ingest import QDRANT_COLLECTION, QDRANT_TIMEOUT, embed_texts


def search(query: str, top_k: int = 5) -> list[dict]:
    """Embed the query, search Qdrant, return top_k chunks with text, doc_name, page, score."""
    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
        timeout=QDRANT_TIMEOUT,
    )
    query_vector = embed_texts([query])[0]
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    )
    return [
        {
            "text": r.payload["text"],
            "doc_name": r.payload["doc_name"],
            "page": r.payload["page"],
            "score": r.score,
        }
        for r in response.points
    ]


if __name__ == "__main__":
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(description="Query auditRAG: retrieve chunks and optionally generate an answer.")
    parser.add_argument("question", nargs="*", help="Question (or leave empty to type when prompted)")
    parser.add_argument("-a", "--answer", action="store_true", help="Generate LLM answer from chunks (needs OPENAI_API_KEY)")
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
        from auditrag.generate import generate_answer
        t1 = time.perf_counter()
        gen = generate_answer(question, results)
        generate_ms = round((time.perf_counter() - t1) * 1000)
        if gen:
            print("--- Answer ---")
            print(gen["answer"])
            print("\n--- Sources ---")
            for s in gen["sources"]:
                print(f"  {s['doc_name']} p{s['page']}")
            print(f"\nLatency: retrieve={retrieve_ms} ms, generate={generate_ms} ms, total={retrieve_ms + generate_ms} ms\n")
        else:
            print("(Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env to generate an answer.)\n")

    for i, r in enumerate(results, 1):
        preview = r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"]
        print(f"--- Chunk {i} (score={r['score']:.4f}, {r['doc_name']} p{r['page']}) ---")
        print(preview)
        print()
