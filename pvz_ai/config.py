from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["groq", "huggingface", "echo"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "pvz-ai"
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/pvz_ai",
        validation_alias="DATABASE_URL",
    )
    auto_create_tables: bool = Field(
        default=False,
        validation_alias="AUTO_CREATE_TABLES",
    )

    llm_provider: ProviderName = Field(default="groq", validation_alias="LLM_PROVIDER")
    llm_base_url: str | None = Field(default=None, validation_alias="LLM_BASE_URL")
    llm_model: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias="LLM_MODEL",
    )
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")
    llm_temperature: float = Field(default=0.3, validation_alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, validation_alias="LLM_MAX_TOKENS")
    history_limit: int = Field(default=20, validation_alias="HISTORY_LIMIT")

    @computed_field
    @property
    def resolved_llm_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url
        if self.llm_provider == "huggingface":
            return "https://router.huggingface.co/v1"
        return "https://api.groq.com/openai/v1"

    @computed_field
    @property
    def resolved_api_key(self) -> str | None:
        if self.llm_provider == "huggingface":
            return self.hf_token
        if self.llm_provider == "groq":
            return self.groq_api_key
        return None

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
