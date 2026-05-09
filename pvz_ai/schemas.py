from typing import Literal

from pydantic import BaseModel, Field

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
