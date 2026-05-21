from types import SimpleNamespace

from pvz_ai.config import Settings
from pvz_ai.llm import LLMRequestOptions, OpenAICompatibleClient


class FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        FakeAsyncOpenAI.calls.append(
            {
                "base_url": self.owner.base_url,
                "model": kwargs["model"],
                "temperature": kwargs["temperature"],
                "top_p": kwargs["top_p"],
                "max_tokens": kwargs["max_tokens"],
            }
        )
        if "groq" in self.owner.base_url:
            raise RuntimeError("groq unavailable")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="huggingface answer"))
            ]
        )


class FakeChat:
    def __init__(self, owner):
        self.completions = FakeCompletions(owner)


class FakeAsyncOpenAI:
    calls = []

    def __init__(self, *, api_key: str, base_url: str) -> None:
        del api_key
        self.base_url = base_url
        self.chat = FakeChat(self)


async def test_openai_client_falls_back_to_huggingface(monkeypatch):
    FakeAsyncOpenAI.calls = []
    monkeypatch.setattr("pvz_ai.llm.AsyncOpenAI", FakeAsyncOpenAI)
    settings = Settings(
        _env_file=None,
        LLM_PROVIDER="groq",
        GROQ_API_KEY="groq-token",
        HF_TOKEN="hf-token",
    )
    client = OpenAICompatibleClient(settings)

    result = await client.complete(
        [{"role": "user", "content": "hello"}],
        LLMRequestOptions(
            provider_mode="auto",
            temperature=0.2,
            top_p=0.8,
            max_tokens=512,
        ),
    )

    assert result.content == "huggingface answer"
    assert result.provider == "huggingface"
    assert result.fallback_used is True
    assert [call["base_url"] for call in FakeAsyncOpenAI.calls] == [
        "https://api.groq.com/openai/v1",
        "https://router.huggingface.co/v1",
    ]
    assert FakeAsyncOpenAI.calls[1]["max_tokens"] == 512
    assert FakeAsyncOpenAI.calls[1]["top_p"] == 0.8


async def test_openai_client_can_force_huggingface(monkeypatch):
    FakeAsyncOpenAI.calls = []
    monkeypatch.setattr("pvz_ai.llm.AsyncOpenAI", FakeAsyncOpenAI)
    settings = Settings(
        _env_file=None,
        LLM_PROVIDER="groq",
        GROQ_API_KEY="groq-token",
        HF_TOKEN="hf-token",
    )
    client = OpenAICompatibleClient(settings)

    result = await client.complete(
        [{"role": "user", "content": "hello"}],
        LLMRequestOptions(provider_mode="huggingface"),
    )

    assert result.provider == "huggingface"
    assert result.fallback_used is False
    assert [call["base_url"] for call in FakeAsyncOpenAI.calls] == [
        "https://router.huggingface.co/v1",
    ]
