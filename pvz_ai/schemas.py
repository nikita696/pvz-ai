from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pvz_ai.llm import LLMRequestOptions

ProviderMode = Literal["auto", "groq", "huggingface", "echo"]


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=12000)
    provider_mode: ProviderMode | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    system_prompt: str | None = Field(default=None, max_length=4000)

    def to_llm_options(self) -> LLMRequestOptions:
        return LLMRequestOptions(
            provider_mode=self.provider_mode or "auto",
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
        )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    provider: str
    model: str
    status: str = "ok"
    fallback_used: bool = False


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any] | None = None
    name: str | None = None

    def as_text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for item in self.content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[OpenAIMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=8192)
    user: str | None = None
    metadata: dict[str, Any] | None = None

    def normalized_messages(self) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.as_text()}
            for message in self.messages
            if message.as_text()
        ]
