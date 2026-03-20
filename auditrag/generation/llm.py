import anthropic
from openai import OpenAI

from auditrag.core.config import get_settings
from auditrag.generation.prompt import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    CONCISE_SYSTEM_PROMPT, CONCISE_USER_TEMPLATE,
    build_context,
)
from auditrag.observability.cost import cost_usd


def _call_openai(
    question: str, context: str, api_key: str, model: str,
    system_prompt: str = SYSTEM_PROMPT, user_template: str = USER_PROMPT_TEMPLATE,
) -> tuple[str, int, int]:
    """Returns (answer_text, input_tokens, output_tokens)."""
    client = OpenAI(api_key=api_key)
    user_msg = user_template.format(context=context, question=question)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    inp = getattr(usage, "prompt_tokens", 0) if usage else 0
    out = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, inp, out


def _call_anthropic(
    question: str, context: str, api_key: str, model: str,
    system_prompt: str = SYSTEM_PROMPT, user_template: str = USER_PROMPT_TEMPLATE,
) -> tuple[str, int, int]:
    """Returns (answer_text, input_tokens, output_tokens)."""
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = user_template.format(context=context, question=question)
    resp = client.messages.create(
        model=model,
        max_tokens=700,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.1,
    )
    parts: list[str] = []
    for block in resp.content:
        if block.type == "text":
            parts.append(block.text)
    text = "\n".join(parts).strip()
    usage = getattr(resp, "usage", None)
    inp = getattr(usage, "input_tokens", 0) if usage else 0
    out = getattr(usage, "output_tokens", 0) if usage else 0
    return text, inp, out


def generate_answer(question: str, chunks: list[dict], mode: str = "default") -> dict | None:
    """Call the configured LLM with question + chunks.

    Args:
        mode: "default" for full answers with citations, "concise" for short direct answers.

    Returns {answer, sources, usage?, cost_usd?} or None if no key.
    """
    settings = get_settings()

    if not chunks:
        return {"answer": "No relevant context was retrieved.", "sources": [], "usage": None, "cost_usd": None}

    context = build_context(chunks)

    if mode == "concise":
        system_prompt = CONCISE_SYSTEM_PROMPT
        user_template = CONCISE_USER_TEMPLATE
    else:
        system_prompt = SYSTEM_PROMPT
        user_template = USER_PROMPT_TEMPLATE

    provider = (getattr(settings, "generation_provider", "openai") or "openai").lower()
    answer_text: str | None = None
    model_used = ""
    input_tokens, output_tokens = 0, 0

    if provider == "anthropic" and settings.anthropic_api_key:
        model_used = getattr(settings, "anthropic_model", "claude-sonnet-4-5-20250929") or "claude-sonnet-4-5-20250929"
        answer_text, input_tokens, output_tokens = _call_anthropic(
            question, context, settings.anthropic_api_key, model_used,
            system_prompt=system_prompt, user_template=user_template,
        )
    elif settings.openai_api_key:
        model_used = "gpt-4o-mini"
        answer_text, input_tokens, output_tokens = _call_openai(
            question, context, settings.openai_api_key, model_used,
            system_prompt=system_prompt, user_template=user_template,
        )

    if answer_text is None:
        return None

    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    cost = cost_usd(model_used, input_tokens, output_tokens)
    sources = [{"doc_name": c["doc_name"], "page": c["page"]} for c in chunks]
    return {
        "answer": answer_text,
        "sources": sources,
        "usage": usage,
        "cost_usd": round(cost, 6) if cost is not None else None,
        "model_used": model_used or None,
    }
