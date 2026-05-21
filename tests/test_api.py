from httpx import ASGITransport, AsyncClient

from pvz_ai.main import create_app
from tests.conftest import FakeLLM


class BrokenLLM:
    provider = "broken"
    model = "broken-model"

    async def complete(self, messages, options=None):
        del options
        del messages
        raise RuntimeError("provider unavailable")


async def test_health_endpoint(test_settings):
    app = create_app(settings=test_settings, llm_client=FakeLLM())

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "pvz-ai",
        "environment": "test",
    }


async def test_chat_endpoint_persists_and_returns_session(
    session_factory,
    test_settings,
    monkeypatch,
):
    from pvz_ai import main

    monkeypatch.setattr(main, "SessionLocal", session_factory)
    app = create_app(settings=test_settings, llm_client=FakeLLM(answer="API answer"))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/chat", json={"message": "Hello"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "API answer"
    assert payload["session_id"]
    assert payload["provider"] == "test"
    assert payload["model"] == "test-model"
    assert payload["fallback_used"] is False


async def test_chat_endpoint_accepts_model_settings(
    session_factory,
    test_settings,
    monkeypatch,
):
    from pvz_ai import main

    monkeypatch.setattr(main, "SessionLocal", session_factory)
    llm = FakeLLM(answer="custom settings answer")
    app = create_app(settings=test_settings, llm_client=llm)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/chat",
            json={
                "message": "Hello",
                "provider_mode": "huggingface",
                "model": "openai/gpt-oss-120b",
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 256,
                "system_prompt": "You are concise.",
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "custom settings answer"
    assert llm.options[0] is not None
    assert llm.options[0].provider_mode == "huggingface"
    assert llm.options[0].max_tokens == 256
    assert llm.messages[0][0] == {"role": "system", "content": "You are concise."}


async def test_chat_endpoint_returns_error_payload_when_provider_fails(
    session_factory,
    test_settings,
    monkeypatch,
):
    from pvz_ai import main

    monkeypatch.setattr(main, "SessionLocal", session_factory)
    app = create_app(settings=test_settings, llm_client=BrokenLLM())

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/chat", json={"message": "Hello"})

    assert response.status_code == 200
    payload = response.json()
    assert (
        payload["answer"] == "The model request failed. Please try again in a moment."
    )
    assert payload["status"] == "error"
    assert payload["provider"] == "broken"


async def test_openai_models_endpoint(test_settings):
    app = create_app(settings=test_settings, llm_client=FakeLLM())

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert any(model["id"] == test_settings.llm_model for model in payload["data"])


async def test_openai_chat_completion_non_streaming(
    session_factory,
    test_settings,
    monkeypatch,
):
    from pvz_ai import main

    monkeypatch.setattr(main, "SessionLocal", session_factory)
    llm = FakeLLM(answer="Open WebUI answer")
    app = create_app(settings=test_settings, llm_client=llm)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-oss-120b:huggingface",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.2,
                "max_tokens": 128,
                "metadata": {"chat_id": "chat-1"},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "openai/gpt-oss-120b:huggingface"
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Open WebUI answer",
    }
    assert payload["pvz_ai"]["session_id"]
    assert llm.options[0].provider_mode == "huggingface"
    assert llm.options[0].model == "openai/gpt-oss-120b"


async def test_openai_chat_completion_streaming(
    session_factory,
    test_settings,
    monkeypatch,
):
    from pvz_ai import main

    monkeypatch.setattr(main, "SessionLocal", session_factory)
    app = create_app(settings=test_settings, llm_client=FakeLLM(answer="stream answer"))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    body = response.text
    assert "chat.completion.chunk" in body
    assert "stream answer" in body
    assert "data: [DONE]" in body
