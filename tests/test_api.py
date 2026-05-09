from httpx import ASGITransport, AsyncClient

from pvz_ai.main import create_app
from tests.conftest import FakeLLM


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
