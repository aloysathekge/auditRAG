"""Model pricing ($ per million tokens). Base Input and Output only; no cache tiers.

Sources: provider pricing pages (OpenAI, Anthropic). Update when they change.
"""

# $ per MTok (million tokens): (input, output). Cache tiers not included.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    # OpenAI — GPT-5
    "gpt-5.2": (1.75, 14.00),
    "gpt-5-2": (1.75, 14.00),
    "gpt-5.2-pro": (21.00, 168.00),
    "gpt-5-2-pro": (21.00, 168.00),
    "gpt-5-mini": (0.25, 2.00),
    # OpenAI — GPT-4
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Anthropic — Claude
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-sonnet-3-7": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-3-5": (0.80, 4.00),
    "claude-haiku-3": (0.25, 1.25),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-opus-3": (15.00, 75.00),
}


def _model_key(model_id: str) -> str:
    """Normalize model id to a key in PRICE_PER_MTOK."""
    m = model_id.lower().strip()
    if m in PRICE_PER_MTOK:
        return m
    if "claude-sonnet-4" in m or "claude-sonnet-4-5" in m:
        return "claude-sonnet-4-5-20250929"
    if "claude-sonnet" in m:
        return "claude-sonnet-4-5-20250929"
    if "claude-opus" in m:
        return "claude-opus-4-6"
    if "claude-haiku" in m:
        return "claude-haiku-4-5"
    if "gpt-5.2-pro" in m or "gpt-5-2-pro" in m:
        return "gpt-5.2-pro"
    if "gpt-5.2" in m or "gpt-5-2" in m:
        return "gpt-5.2"
    if "gpt-5-mini" in m:
        return "gpt-5-mini"
    if "gpt-5" in m:
        return "gpt-5.2"
    if "gpt-4o-mini" in m:
        return "gpt-4o-mini"
    if "gpt-4o" in m:
        return "gpt-4o"
    return ""


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost in USD for the given token counts. Returns None if model has no price."""
    key = _model_key(model_id)
    if not key:
        return None
    inp_price, out_price = PRICE_PER_MTOK[key]
    return (input_tokens / 1_000_000) * inp_price + (output_tokens / 1_000_000) * out_price
