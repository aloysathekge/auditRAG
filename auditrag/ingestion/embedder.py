import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from auditrag.core.config import get_settings

QDRANT_COLLECTION = "auditrag_chunks"
QDRANT_UPSERT_BATCH_SIZE = 50
QDRANT_TIMEOUT = 120

OPENAI_EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 100

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading local model: {LOCAL_EMBED_MODEL}")
        _local_model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _local_model


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _get_local_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        print(f"  Embedded batch {i // EMBED_BATCH_SIZE + 1} ({len(batch)} chunks)")
    return all_embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add an 'embedding' field to each chunk. Uses local or OpenAI per EMBEDDING_PROVIDER."""
    settings = get_settings()
    texts = [c["text"] for c in chunks]
    print(f"  Provider: {settings.embedding_provider}")
    if settings.embedding_provider == "openai":
        all_embeddings = _embed_openai(texts)
    else:
        all_embeddings = _embed_local(texts)
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts (e.g. a query). Uses same provider as embed_chunks."""
    settings = get_settings()
    if not texts:
        return []
    if settings.embedding_provider == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


def _get_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
        timeout=QDRANT_TIMEOUT,
    )


def doc_already_ingested(client: QdrantClient, doc_name: str, knowledge_base: str | None = None) -> bool:
    """Return True if at least one chunk for this doc_name exists in Qdrant."""
    try:
        conditions = [FieldCondition(key="doc_name", match=MatchValue(value=doc_name))]
        if knowledge_base is not None:
            conditions.append(FieldCondition(key="knowledge_base", match=MatchValue(value=knowledge_base)))
        result, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=conditions),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(result) > 0
    except Exception:
        return False


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    """Create the auditrag collection if it does not exist."""
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    if QDRANT_COLLECTION not in names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"  Created collection '{QDRANT_COLLECTION}' (size={vector_size})")
    else:
        print(f"  Using existing collection '{QDRANT_COLLECTION}'")


def upsert_chunks_to_qdrant(chunks: list[dict], skip_if_doc_exists: bool = True) -> int:
    """Upload embedded chunks to Qdrant. Returns count upserted (0 if skipped)."""
    if not chunks or "embedding" not in chunks[0]:
        raise ValueError("Chunks must be embedded first (call embed_chunks)")
    client = _get_client()
    doc_name = chunks[0]["doc_name"]
    knowledge_base = chunks[0].get("knowledge_base")
    if skip_if_doc_exists and doc_already_ingested(client, doc_name, knowledge_base):
        print(f"  Skipping upsert: '{doc_name}' already in Qdrant (use skip_if_doc_exists=False to re-ingest).")
        return 0
    vector_size = len(chunks[0]["embedding"])
    ensure_collection(client, vector_size)

    points = []
    for c in chunks:
        point_id = int(hashlib.md5(c["chunk_id"].encode()).hexdigest()[:16], 16)
        payload = {
            "chunk_id": c["chunk_id"],
            "doc_name": c["doc_name"],
            "page": c["page"],
            "text": c["text"],
            "knowledge_base": c.get("knowledge_base", "default"),
        }
        points.append(
            PointStruct(
                id=point_id,
                vector=c["embedding"],
                payload=payload,
            )
        )

    total = 0
    for i in range(0, len(points), QDRANT_UPSERT_BATCH_SIZE):
        batch = points[i : i + QDRANT_UPSERT_BATCH_SIZE]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        total += len(batch)
        print(f"  Upserted batch {i // QDRANT_UPSERT_BATCH_SIZE + 1} ({total}/{len(points)} points)")
    return total


def delete_doc_from_qdrant(doc_name: str, knowledge_base: str = "default") -> int:
    """Delete all chunks for a doc_name + knowledge_base. Returns count deleted."""
    client = _get_client()
    try:
        # Count before delete
        conditions = [
            FieldCondition(key="doc_name", match=MatchValue(value=doc_name)),
            FieldCondition(key="knowledge_base", match=MatchValue(value=knowledge_base)),
        ]
        scroll_filter = Filter(must=conditions)
        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            return 0

        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=scroll_filter,
        )
        return -1  # Qdrant delete doesn't return count; -1 signals success
    except Exception:
        return 0


def list_docs_in_qdrant(knowledge_base: str | None = None) -> list[dict]:
    """List unique documents in Qdrant, optionally filtered by knowledge_base."""
    client = _get_client()
    try:
        conditions = []
        if knowledge_base is not None:
            conditions.append(FieldCondition(key="knowledge_base", match=MatchValue(value=knowledge_base)))
        scroll_filter = Filter(must=conditions) if conditions else None

        doc_counts: dict[str, dict] = {}
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=scroll_filter,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in result:
                payload = pt.payload or {}
                doc_name = payload.get("doc_name", "")
                kb = payload.get("knowledge_base", "default")
                key = f"{kb}::{doc_name}"
                if key not in doc_counts:
                    doc_counts[key] = {"doc_name": doc_name, "knowledge_base": kb, "chunks_count": 0}
                doc_counts[key]["chunks_count"] += 1
            if next_offset is None:
                break
            offset = next_offset

        return list(doc_counts.values())
    except Exception:
        return []
