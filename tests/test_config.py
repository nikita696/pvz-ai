from pvz_ai.config import Settings
from pvz_ai.database import normalize_database_url


def test_default_database_url_is_local_sqlite_fallback(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.database_url == "sqlite+aiosqlite:///./data/pvz_ai.sqlite"
    assert settings.uses_sqlite_fallback is True


def test_vercel_database_fallback_uses_tmp(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("VERCEL", "1")

    settings = Settings(_env_file=None)

    assert settings.database_url == "sqlite+aiosqlite:////tmp/pvz_ai.sqlite"
    assert settings.uses_sqlite_fallback is True


def test_groq_defaults_to_openai_compatible_endpoint():
    settings = Settings(
        LLM_PROVIDER="groq",
        LLM_MODEL="openai/gpt-oss-120b",
        GROQ_API_KEY="groq-key",
    )

    assert settings.llm_provider == "groq"
    assert settings.resolved_llm_base_url == "https://api.groq.com/openai/v1"
    assert settings.groq_base_url == "https://api.groq.com/openai/v1"
    assert settings.resolved_api_key == "groq-key"
    assert settings.llm_model == "openai/gpt-oss-120b"
    assert settings.llm_top_p == 1.0


def test_huggingface_uses_hf_token_and_router():
    settings = Settings(LLM_PROVIDER="huggingface", HF_TOKEN="hf-token")

    assert settings.resolved_llm_base_url == "https://router.huggingface.co/v1"
    assert settings.hf_base_url == "https://router.huggingface.co/v1"
    assert settings.resolved_api_key == "hf-token"


def test_legacy_postgres_url_is_normalized_for_asyncpg():
    assert (
        normalize_database_url("postgres://user:pass@host/db")
        == "postgresql+asyncpg://user:pass@host/db"
    )
    assert (
        normalize_database_url("postgresql://user:pass@host/db")
        == "postgresql+asyncpg://user:pass@host/db"
    )


def test_neon_url_query_is_normalized_for_asyncpg():
    assert (
        normalize_database_url(
            "postgresql://user:pass@host/db?sslmode=require&channel_binding=require"
        )
        == "postgresql+asyncpg://user:pass@host/db?ssl=require"
    )
