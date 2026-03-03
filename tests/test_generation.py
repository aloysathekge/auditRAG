from unittest.mock import MagicMock, patch

from auditrag.generation.llm import generate_answer


class DummySettings:
    def __init__(
        self,
        generation_provider: str = "openai",
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        anthropic_model: str | None = None,
    ):
        self.generation_provider = generation_provider
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.anthropic_model = anthropic_model


def _chunks():
    return [{"doc_name": "doc", "page": 1, "text": "some financial context"}]


def test_generate_answer_handles_empty_chunks_gracefully():
    with patch("auditrag.generation.llm.get_settings", return_value=DummySettings()):
        result = generate_answer("What is revenue?", [])

    assert result is not None
    assert "answer" in result
    assert result["sources"] == []
    assert result["cost_usd"] is None


def test_generate_answer_returns_none_when_no_api_keys():
    # Non-empty chunks but no OpenAI/Anthropic keys configured
    with patch("auditrag.generation.llm.get_settings", return_value=DummySettings()):
        result = generate_answer("What is revenue?", _chunks())

    assert result is None


def test_generate_answer_calls_openai_when_key_present():
    fake_settings = DummySettings(openai_api_key="fake-key")

    fake_usage = MagicMock()
    fake_usage.prompt_tokens = 10
    fake_usage.completion_tokens = 5

    fake_choice = MagicMock()
    fake_choice.message.content = "The revenue was 100 million."

    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = fake_usage

    with (
        patch("auditrag.generation.llm.get_settings", return_value=fake_settings),
        patch("auditrag.generation.llm.OpenAI") as mock_openai_cls,
        patch("auditrag.generation.llm.cost_usd", return_value=0.001),
        patch("auditrag.generation.llm.build_context", return_value="CTX"),
    ):
        mock_openai_cls.return_value.chat.completions.create.return_value = fake_response

        result = generate_answer("What is revenue?", _chunks())

    assert result is not None
    assert result["answer"].startswith("The revenue")
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 5
    assert result["cost_usd"] == 0.001
    assert result["model_used"] == "gpt-4o-mini"


def test_generate_answer_calls_anthropic_when_configured():
    fake_settings = DummySettings(
        generation_provider="anthropic",
        anthropic_api_key="anthropic-key",
        anthropic_model="claude-sonnet-4-5-20250929",
    )

    fake_usage = MagicMock()
    fake_usage.input_tokens = 12
    fake_usage.output_tokens = 7

    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "Anthropic answer."

    fake_resp = MagicMock()
    fake_resp.content = [fake_block]
    fake_resp.usage = fake_usage

    with (
        patch("auditrag.generation.llm.get_settings", return_value=fake_settings),
        patch("auditrag.generation.llm.anthropic.Anthropic") as mock_anthropic_cls,
        patch("auditrag.generation.llm.cost_usd", return_value=0.002),
        patch("auditrag.generation.llm.build_context", return_value="CTX"),
    ):
        mock_anthropic_cls.return_value.messages.create.return_value = fake_resp

        result = generate_answer("What is revenue?", _chunks())

    assert result is not None
    assert result["answer"].startswith("Anthropic")
    assert result["usage"]["input_tokens"] == 12
    assert result["usage"]["output_tokens"] == 7
    assert result["cost_usd"] == 0.002
    assert result["model_used"] == "claude-sonnet-4-5-20250929"

