from typing import Protocol

from openai import AsyncOpenAI

from pvz_ai.config import Settings

ChatPayload = list[dict[str, str]]


class LLMConfigurationError(RuntimeError):
    pass


class LLMClient(Protocol):
    provider: str
    model: str

    async def complete(self, messages: ChatPayload) -> str:
        raise NotImplementedError


class OpenAICompatibleClient:
    def __init__(self, settings: Settings) -> None:
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self._api_key = settings.resolved_api_key
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if not self._api_key:
            raise LLMConfigurationError(
                f"{self.provider} is selected but its API token is not configured."
            )
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._settings.resolved_llm_base_url,
            )
        return self._client

    async def complete(self, messages: ChatPayload) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response.")
        return content


class EchoLLMClient:
    provider = "echo"
    model = "echo-local"

    async def complete(self, messages: ChatPayload) -> str:
        user_messages = [item["content"] for item in messages if item["role"] == "user"]
        latest = user_messages[-1] if user_messages else ""
        return f"Echo: {latest}"


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "echo":
        return EchoLLMClient()
    return OpenAICompatibleClient(settings)
