import logging
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pvz_ai.config import Settings
from pvz_ai.llm import LLMClient, LLMConfigurationError
from pvz_ai.models import ChatMessage, ConversationSession, utc_now

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are pvz-ai, a concise, helpful AI assistant for operational PVZ work. "
    "Answer clearly and preserve useful context from the conversation."
)


class ChatProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatTurn:
    session_id: str
    answer: str
    provider: str
    model: str
    status: str = "ok"


class ChatService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        llm_client: LLMClient,
        settings: Settings,
    ) -> None:
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.settings = settings

    async def send_message(
        self,
        message: str,
        session_id: str | None = None,
        *,
        raise_on_error: bool = True,
    ) -> ChatTurn:
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Message cannot be empty.")

        start = time.perf_counter()
        async with self.session_factory() as db:
            session = await self._get_or_create_session(db, session_id, cleaned_message)
            await self._store_message(
                db,
                session_id=session.id,
                role="user",
                content=cleaned_message,
            )
            await db.flush()

            messages = await self._build_llm_messages(db, session.id)

            try:
                answer = await self.llm_client.complete(messages)
            except LLMConfigurationError as exc:
                answer = (
                    "LLM provider is not configured yet. "
                    "Set GROQ_API_KEY for Groq or HF_TOKEN for Hugging Face."
                )
                return await self._handle_llm_error(
                    db,
                    session.id,
                    answer,
                    exc,
                    raise_on_error,
                )
            except Exception as exc:
                answer = "The model request failed. Please try again in a moment."
                return await self._handle_llm_error(
                    db,
                    session.id,
                    answer,
                    exc,
                    raise_on_error,
                )

            await self._store_message(
                db,
                session_id=session.id,
                role="assistant",
                content=answer,
                provider=self.llm_client.provider,
                model=self.llm_client.model,
            )
            session.updated_at = utc_now()
            await db.commit()

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "chat completed provider=%s model=%s elapsed_ms=%s",
                self.llm_client.provider,
                self.llm_client.model,
                elapsed_ms,
                extra={"session_id": session.id},
            )
            return ChatTurn(
                session_id=session.id,
                answer=answer,
                provider=self.llm_client.provider,
                model=self.llm_client.model,
            )

    async def _get_or_create_session(
        self,
        db: AsyncSession,
        session_id: str | None,
        first_message: str,
    ) -> ConversationSession:
        if session_id:
            existing = await db.get(ConversationSession, session_id)
            if existing:
                return existing

        title = (
            first_message[:157] + "..." if len(first_message) > 160 else first_message
        )
        session = ConversationSession(
            id=session_id or None,
            title=title,
            provider=self.llm_client.provider,
            model=self.llm_client.model,
        )
        db.add(session)
        await db.flush()
        return session

    async def _store_message(
        self,
        db: AsyncSession,
        *,
        session_id: str,
        role: str,
        content: str,
        status: str = "ok",
        provider: str | None = None,
        model: str | None = None,
        error_message: str | None = None,
    ) -> ChatMessage:
        chat_message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            status=status,
            provider=provider,
            model=model,
            error_message=error_message,
        )
        db.add(chat_message)
        return chat_message

    async def _build_llm_messages(
        self,
        db: AsyncSession,
        session_id: str,
    ) -> list[dict[str, str]]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(self.settings.history_limit)
        )
        result = await db.execute(stmt)
        stored_messages = list(reversed(result.scalars().all()))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(
            {"role": item.role, "content": item.content}
            for item in stored_messages
            if item.status == "ok" and item.role in {"user", "assistant"}
        )
        return messages

    async def _handle_llm_error(
        self,
        db: AsyncSession,
        session_id: str,
        answer: str,
        exc: Exception,
        raise_on_error: bool,
    ) -> ChatTurn:
        await self._store_message(
            db,
            session_id=session_id,
            role="assistant",
            content=answer,
            status="error",
            provider=self.llm_client.provider,
            model=self.llm_client.model,
            error_message=type(exc).__name__,
        )
        await db.commit()

        logger.warning(
            "chat provider failed provider=%s model=%s error=%s",
            self.llm_client.provider,
            self.llm_client.model,
            type(exc).__name__,
            extra={"session_id": session_id},
        )

        if raise_on_error:
            raise ChatProviderError(answer) from exc

        return ChatTurn(
            session_id=session_id,
            answer=answer,
            provider=self.llm_client.provider,
            model=self.llm_client.model,
            status="error",
        )
