from sqlalchemy import select

from pvz_ai.models import ChatMessage, ConversationSession
from pvz_ai.services import ChatService
from tests.conftest import FakeLLM


async def test_chat_service_creates_session_and_persists_history(
    session_factory,
    test_settings,
):
    llm = FakeLLM(answer="Привет, Ник")
    service = ChatService(session_factory, llm, test_settings)

    turn = await service.send_message("Привет")

    assert turn.answer == "Привет, Ник"
    assert turn.session_id
    assert llm.messages[0][-1] == {"role": "user", "content": "Привет"}

    async with session_factory() as db:
        sessions = (await db.execute(select(ConversationSession))).scalars().all()
        messages = (
            (await db.execute(select(ChatMessage).order_by(ChatMessage.id)))
            .scalars()
            .all()
        )

    assert len(sessions) == 1
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].provider == "test"
    assert messages[1].model == "test-model"


async def test_chat_service_reuses_existing_session(session_factory, test_settings):
    service = ChatService(session_factory, FakeLLM(answer="ok"), test_settings)

    first = await service.send_message("one")
    second = await service.send_message("two", session_id=first.session_id)

    assert second.session_id == first.session_id

    async with session_factory() as db:
        messages = (
            (await db.execute(select(ChatMessage).order_by(ChatMessage.id)))
            .scalars()
            .all()
        )

    assert [message.content for message in messages] == ["one", "ok", "two", "ok"]
