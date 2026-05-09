from pvz_ai.config import Settings
from pvz_ai.database import normalize_database_url


def test_groq_defaults_to_openai_compatible_endpoint():
    settings = Settings(GROQ_API_KEY="groq-key")

    assert settings.llm_provider == "groq"
    assert settings.resolved_llm_base_url == "https://api.groq.com/openai/v1"
    assert settings.resolved_api_key == "groq-key"
    assert settings.llm_model == "openai/gpt-oss-120b"


def test_huggingface_uses_hf_token_and_router():
    settings = Settings(LLM_PROVIDER="huggingface", HF_TOKEN="hf-token")

    assert settings.resolved_llm_base_url == "https://router.huggingface.co/v1"
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
