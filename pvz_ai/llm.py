import logging
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from openai import AsyncOpenAI

from pvz_ai.config import Settings

ChatPayload = list[dict[str, str]]
ProviderMode = Literal["auto", "groq", "huggingface", "echo"]

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMRequestOptions:
    provider_mode: ProviderMode = "auto"
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None


@dataclass(frozen=True)
class LLMResult:
    content: str
    provider: str
    model: str
    fallback_used: bool = False


class LLMClient(Protocol):
    provider: str
    model: str

    async def complete(
        self,
        messages: ChatPayload,
        options: LLMRequestOptions | None = None,
    ) -> LLMResult:
        raise NotImplementedError


class OpenAICompatibleClient:
    def __init__(self, settings: Settings) -> None:
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self._settings = settings
        self._clients: dict[str, AsyncOpenAI] = {}

    def _provider_order(self, mode: ProviderMode) -> list[str]:
        if mode != "auto":
            return [mode]

        order = [self.provider]
        if self.provider == "groq" and self._settings.hf_token:
            order.append("huggingface")
        elif self.provider == "huggingface" and self._settings.groq_api_key:
            order.append("groq")
        return list(dict.fromkeys(order))

    def _base_url_for(self, provider: str) -> str:
        if provider == self._settings.llm_provider and self._settings.llm_base_url:
            return self._settings.llm_base_url
        if provider == "huggingface":
            return self._settings.hf_base_url
        if provider == "groq":
            return self._settings.groq_base_url
        raise LLMConfigurationError(f"Unsupported LLM provider: {provider}.")

    def _api_key_for(self, provider: str) -> str | None:
        if provider == "huggingface":
            return self._settings.hf_token
        if provider == "groq":
            return self._settings.groq_api_key
        return None

    def _client_for(self, provider: str) -> AsyncOpenAI:
        api_key = self._api_key_for(provider)
        if not api_key:
            raise LLMConfigurationError(
                f"{provider} is selected but its API token is not configured."
            )

        base_url = self._base_url_for(provider)
        cache_key = f"{provider}:{base_url}"
        if cache_key not in self._clients:
            self._clients[cache_key] = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        return self._clients[cache_key]

    async def _complete_echo(
        self,
        messages: ChatPayload,
        options: LLMRequestOptions,
    ) -> LLMResult:
        del options
        user_messages = [item["content"] for item in messages if item["role"] == "user"]
        latest = user_messages[-1] if user_messages else ""
        return LLMResult(
            content=f"Echo: {latest}",
            provider="echo",
            model="echo-local",
        )

    async def _complete_provider(
        self,
        provider: str,
        messages: ChatPayload,
        options: LLMRequestOptions,
    ) -> LLMResult:
        if provider == "echo":
            return await self._complete_echo(messages, options)

        model = options.model or self.model
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": (
                options.temperature
                if options.temperature is not None
                else self._settings.llm_temperature
            ),
            "max_tokens": (
                options.max_tokens
                if options.max_tokens is not None
                else self._settings.llm_max_tokens
            ),
        }
        top_p = options.top_p if options.top_p is not None else self._settings.llm_top_p
        if top_p is not None:
            request_kwargs["top_p"] = top_p

        response = await self._client_for(provider).chat.completions.create(
            **request_kwargs,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response.")

        return LLMResult(
            content=content,
            provider=provider,
            model=model,
        )

    async def complete(
        self,
        messages: ChatPayload,
        options: LLMRequestOptions | None = None,
    ) -> LLMResult:
        options = options or LLMRequestOptions()
        providers = self._provider_order(options.provider_mode)
        last_error: Exception | None = None

        for attempt_index, provider in enumerate(providers):
            start = time.perf_counter()
            model = options.model or self.model
            try:
                result = await self._complete_provider(provider, messages, options)
            except Exception as exc:
                last_error = exc
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                logger.warning(
                    "llm attempt failed provider=%s model=%s "
                    "status_code=%s error=%s elapsed_ms=%s",
                    provider,
                    model,
                    getattr(exc, "status_code", "-"),
                    type(exc).__name__,
                    elapsed_ms,
                )
                continue

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            fallback_used = attempt_index > 0
            logger.info(
                "llm attempt completed provider=%s model=%s fallback_used=%s "
                "elapsed_ms=%s",
                result.provider,
                result.model,
                fallback_used,
                elapsed_ms,
            )
            return LLMResult(
                content=result.content,
                provider=result.provider,
                model=result.model,
                fallback_used=fallback_used,
            )

        if last_error:
            raise last_error
        raise LLMConfigurationError("No LLM provider is configured.")


class EchoLLMClient:
    provider = "echo"
    model = "echo-local"

    async def complete(
        self,
        messages: ChatPayload,
        options: LLMRequestOptions | None = None,
    ) -> LLMResult:
        del options
        user_messages = [item["content"] for item in messages if item["role"] == "user"]
        latest = user_messages[-1] if user_messages else ""
        return LLMResult(
            content=f"Echo: {latest}",
            provider=self.provider,
            model=self.model,
        )


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "echo":
        return EchoLLMClient()
    return OpenAICompatibleClient(settings)
