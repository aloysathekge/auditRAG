"""Generate an answer from retrieved chunks using an LLM."""

from openai import OpenAI

from auditrag.config import get_settings

SYSTEM_PROMPT = """You answer questions using only the provided context from SEC filings (10-K, 10-Q).
- Base your answer only on the context below. Do not use outside knowledge.
- If the context does not contain enough information, say so clearly.
- When you use a number or fact, cite the source as [doc_name, page X].
- Keep the answer concise and factual."""

USER_PROMPT_TEMPLATE = """Context from filings:

{context}

Question: {question}

Answer (cite sources as [doc_name, page N]):"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] ({c['doc_name']}, page {c['page']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]) -> dict | None:
    """Call the LLM with question + chunks. Returns {answer, sources} or None if no API key."""
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    if not chunks:
        return {"answer": "No relevant context was retrieved.", "sources": []}

    context = _build_context(chunks)
    user_msg = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    answer_text = response.choices[0].message.content or ""

    sources = [{"doc_name": c["doc_name"], "page": c["page"]} for c in chunks]
    return {"answer": answer_text.strip(), "sources": sources}
