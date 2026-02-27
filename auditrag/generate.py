"""Generate an answer from retrieved chunks using an LLM (OpenAI or Anthropic)."""

from openai import OpenAI
import anthropic

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


def _call_openai(question: str, context: str, api_key: str, model:str) -> str:
    client = OpenAI(api_key=api_key)
    user_msg = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return (response.choices[0].message.content or "").strip()


def _call_anthropic(question: str, context: str, api_key: str, model: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    resp = client.messages.create(
        model=model,
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.1,
    )
    parts: list[str] = []
    for block in resp.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def generate_answer(question: str, chunks: list[dict]) -> dict | None:
    """Call the configured LLM with question + chunks. Returns {answer, sources} or None if no key."""
    settings = get_settings()

    if not chunks:
        return {"answer": "No relevant context was retrieved.", "sources": []}

    context = _build_context(chunks)

    provider = (getattr(settings, "generation_provider", "openai") or "openai").lower()
    answer_text: str | None = None

    if provider == "anthropic" and settings.anthropic_api_key:
        model = getattr(settings, "anthropic_model", "claude-sonnet-4-5-20250929") or "claude-sonnet-4-5-20250929"
        answer_text = _call_anthropic(question, context, settings.anthropic_api_key, model)
    elif settings.openai_api_key:
        # Default: OpenAI if key is present
        answer_text = _call_openai(question, context, settings.openai_api_key)

    if answer_text is None:
        # No usable key configured
        return None

    sources = [{"doc_name": c["doc_name"], "page": c["page"]} for c in chunks]
    return {"answer": answer_text, "sources": sources}
