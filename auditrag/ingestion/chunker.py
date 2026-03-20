def chunk_pages(
    pages: list[dict],
    doc_name: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    knowledge_base: str = "default",
) -> list[dict]:
    """Split extracted pages into overlapping chunks with metadata.

    Uses a sliding window in words. chunk_size and chunk_overlap are in words.
    """
    full_text = ""
    page_boundaries: list[tuple[int, int]] = []

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
            "chunk_id": f"{knowledge_base}_{doc_name}_chunk_{len(chunks)}",
            "doc_name": doc_name,
            "page": source_page,
            "text": chunk_text,
            "knowledge_base": knowledge_base,
        })

        idx += chunk_size - chunk_overlap

    return chunks
