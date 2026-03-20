SYSTEM_PROMPT = """You answer questions using only the provided context from the knowledge base.
- Base your answer only on the context below. Do not use outside knowledge.
- If the context does not contain enough information, say so clearly.
- When you use a number or fact, cite the source as [doc_name, page X].
- Keep the answer concise and factual."""

CONCISE_SYSTEM_PROMPT = """You answer questions using only the provided context.
- Base your answer only on the context below. Do not use outside knowledge.
- Respond with ONLY the direct answer — no preamble, no citations, no markdown formatting, no extra explanation.
- If the answer is a name, number, or short phrase, return just that.
- Example: "Seth Weidman" not "The author is **Seth Weidman** [source, page 1]."."""

USER_PROMPT_TEMPLATE = """Context from knowledge base:

{context}

Question: {question}

Answer (cite sources as [doc_name, page N]):"""

CONCISE_USER_TEMPLATE = """Context:

{context}

Question: {question}

Answer:"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] ({c['doc_name']}, page {c['page']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)
