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
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ").strip()
    if not question:
        print("No question given.")
        sys.exit(1)

    print("Searching...")
    results = search(question, top_k=5)
    print(f"Found {len(results)} chunks\n")
    for i, r in enumerate(results, 1):
        preview = r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"]
        print(f"--- Chunk {i} (score={r['score']:.4f}, {r['doc_name']} p{r['page']}) ---")
        print(preview)
        print()
