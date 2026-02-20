from pathlib import Path

import httpx
import pdfplumber
from datasets import load_dataset

from auditrag.config import get_settings

DATA_DIR = Path("data/financebench/pdfs")


def load_financebench():
    """Load the FinanceBench dataset from HuggingFace."""
    ds = load_dataset("PatronusAI/financebench", split="train")
    return ds


def download_pdf(url: str, doc_name: str) -> Path:
    """Download a PDF if not already cached locally. Returns the file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{doc_name}.pdf"
    if path.exists():
        print(f"  Already cached: {path}")
        return path

    print(f"  Downloading: {url}")
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"  Saved: {path}")
    return path


def extract_text(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns list of {page, text}."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(
    pages: list[dict],
    doc_name: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split extracted pages into overlapping chunks with metadata.

    Uses a character-based sliding window. chunk_size and chunk_overlap are in
    words (split on whitespace) to roughly approximate token counts.
    """
    full_text = ""
    page_boundaries: list[tuple[int, int]] = []  # (char_start, page_number)

    for page_info in pages:
        start = len(full_text)
        full_text += page_info["text"] + "\n"
        page_boundaries.append((start, page_info["page"]))

    words = full_text.split()
    chunks = []
    idx = 0

    while idx < len(words):
        chunk_words = words[idx : idx + chunk_size]
        chunk_text = " ".join(chunk_words)

        char_start = full_text.index(chunk_words[0], max(0, idx - 1))
        source_page = 1
        for boundary_start, page_num in page_boundaries:
            if boundary_start <= char_start:
                source_page = page_num

        chunks.append({
            "chunk_id": f"{doc_name}_chunk_{len(chunks)}",
            "doc_name": doc_name,
            "page": source_page,
            "text": chunk_text,
        })

        idx += chunk_size - chunk_overlap

    return chunks


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
    """Add an 'embedding' field to each chunk.

    Uses local model (bge-small) or OpenAI depending on EMBEDDING_PROVIDER setting.
    """
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


if __name__ == "__main__":
    ds = load_financebench()
    print(f"Total records: {len(ds)}")

    # Pick the first record as a demo
    sample = ds[0]
    doc_name = sample["doc_name"]
    doc_link = sample["doc_link"]
    print(f"\nDoc: {doc_name}")
    print(f"Link: {doc_link}")

    # Step 1: Download
    pdf_path = download_pdf(doc_link, doc_name)

    # Step 2: Extract text
    pages = extract_text(pdf_path)
    print(f"\nExtracted {len(pages)} pages with text")

    # Step 3: Chunk
    chunks = chunk_pages(pages, doc_name)
    print(f"Created {len(chunks)} chunks")

    # Step 4: Embed (only first 3 chunks as demo to save cost)
    demo_chunks = chunks[:3]
    demo_chunks = embed_chunks(demo_chunks)
    print(f"\nEmbedding dimension: {len(demo_chunks[0]['embedding'])}")
    print(f"First 5 values: {demo_chunks[0]['embedding'][:5]}")
