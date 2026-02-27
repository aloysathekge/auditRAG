SYSTEM_PROMPT = """You answer questions using only the provided context from SEC filings (10-K, 10-Q).
- Base your answer only on the context below. Do not use outside knowledge.
- If the context does not contain enough information, say so clearly.
- When you use a number or fact, cite the source as [doc_name, page X].
- Keep the answer concise and factual."""

USER_PROMPT_TEMPLATE = """Context from filings:

{context}

Question: {question}

Answer (cite sources as [doc_name, page N]):"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] ({c['doc_name']}, page {c['page']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)
