"""Generate Q&A pairs from document chunks using the configured LLM."""
import json
import logging

from openai import OpenAI
import anthropic

from auditrag.core.config import get_settings

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are a Q&A dataset generator. Given a text chunk from a document, generate question-answer pairs that test comprehension of the key facts in the text.

Rules:
- Generate 2-3 question-answer pairs per chunk.
- Questions should be answerable ONLY from the given text.
- Answers should be concise and factual, directly quoted or paraphrased from the text.
- Return valid JSON: a list of objects with "question" and "answer" keys.
- Do NOT include any text outside the JSON array.

Example output:
[
  {"question": "What was the total revenue?", "answer": "$5.2 billion"},
  {"question": "Who is the CEO?", "answer": "Jane Smith"}
]"""

QA_USER_TEMPLATE = """Document: {doc_name} (page {page})

Text:
{text}

Generate Q&A pairs as a JSON array:"""


def _parse_qa_json(raw: str) -> list[dict]:
    """Best-effort parse of LLM JSON output."""
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        pairs = json.loads(raw)
        if isinstance(pairs, list):
            return [p for p in pairs if isinstance(p, dict) and "question" in p and "answer" in p]
    except json.JSONDecodeError:
        logger.warning("Failed to parse Q&A JSON from LLM output")
    return []


def generate_qa_pairs(chunks: list[dict], sample_every: int = 3) -> list[dict]:
    """Generate Q&A pairs from a sample of chunks.

    Args:
        chunks: List of chunk dicts with keys: chunk_id, doc_name, page, text, knowledge_base.
        sample_every: Generate Q&A for every Nth chunk to control cost. Default 3.

    Returns:
        List of dicts with keys: chunk_id, doc_name, knowledge_base, question, answer.
    """
    settings = get_settings()
    provider = (getattr(settings, "generation_provider", "openai") or "openai").lower()

    sampled = chunks[::sample_every]
    if not sampled:
        return []

    all_pairs: list[dict] = []

    for chunk in sampled:
        user_msg = QA_USER_TEMPLATE.format(
            doc_name=chunk["doc_name"],
            page=chunk.get("page", "?"),
            text=chunk["text"][:3000],  # cap to avoid token overflow
        )

        try:
            if provider == "anthropic" and settings.anthropic_api_key:
                model = getattr(settings, "anthropic_model", "claude-sonnet-4-5-20250929") or "claude-sonnet-4-5-20250929"
                client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                resp = client.messages.create(
                    model=model,
                    max_tokens=500,
                    system=QA_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                    temperature=0.3,
                )
                raw = "".join(b.text for b in resp.content if b.type == "text")
            elif settings.openai_api_key:
                client = OpenAI(api_key=settings.openai_api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": QA_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.3,
                )
                raw = (resp.choices[0].message.content or "").strip()
            else:
                logger.warning("No LLM API key configured — skipping Q&A generation")
                return []

            pairs = _parse_qa_json(raw)
            for p in pairs:
                all_pairs.append({
                    "chunk_id": chunk["chunk_id"],
                    "doc_name": chunk["doc_name"],
                    "knowledge_base": chunk.get("knowledge_base", "default"),
                    "question": p["question"],
                    "answer": p["answer"],
                })
        except Exception:
            logger.exception("Failed to generate Q&A for chunk %s", chunk.get("chunk_id"))
            continue

    logger.info("Generated %d Q&A pairs from %d chunks", len(all_pairs), len(sampled))
    return all_pairs
