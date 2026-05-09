import logging
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pvz_ai.config import Settings
from pvz_ai.llm import LLMClient, LLMConfigurationError, LLMRequestOptions
from pvz_ai.logging_config import emit_structured_log
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
    fallback_used: bool = False


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
        model_options: LLMRequestOptions | None = None,
        raise_on_error: bool = True,
    ) -> ChatTurn:
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Message cannot be empty.")

        model_options = model_options or LLMRequestOptions()
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

            messages = await self._build_llm_messages(
                db,
                session.id,
                system_prompt=model_options.system_prompt,
            )

            try:
                result = await self.llm_client.complete(messages, model_options)
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
                    model_options,
                    raise_on_error,
                )
            except Exception as exc:
                answer = "The model request failed. Please try again in a moment."
                return await self._handle_llm_error(
                    db,
                    session.id,
                    answer,
                    exc,
                    model_options,
                    raise_on_error,
                )

            await self._store_message(
                db,
                session_id=session.id,
                role="assistant",
                content=result.content,
                provider=result.provider,
                model=result.model,
            )
            session.provider = result.provider
            session.model = result.model
            session.updated_at = utc_now()
            await db.commit()

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            emit_structured_log(
                logger,
                logging.INFO,
                "chat_completed",
                provider=result.provider,
                model=result.model,
                fallback_used=result.fallback_used,
                elapsed_ms=elapsed_ms,
                session_id=session.id,
            )
            return ChatTurn(
                session_id=session.id,
                answer=result.content,
                provider=result.provider,
                model=result.model,
                fallback_used=result.fallback_used,
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
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(self.settings.history_limit)
        )
        result = await db.execute(stmt)
        stored_messages = list(reversed(result.scalars().all()))

        prompt = (
            system_prompt.strip()
            if system_prompt and system_prompt.strip()
            else SYSTEM_PROMPT
        )
        messages = [{"role": "system", "content": prompt}]
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
        model_options: LLMRequestOptions,
        raise_on_error: bool,
    ) -> ChatTurn:
        provider, model = self._requested_provider_model(model_options)
        await self._store_message(
            db,
            session_id=session_id,
            role="assistant",
            content=answer,
            status="error",
            provider=provider,
            model=model,
            error_message=type(exc).__name__,
        )
        await db.commit()

        emit_structured_log(
            logger,
            logging.WARNING,
            "chat_provider_failed",
            provider=provider,
            model=model,
            error=type(exc).__name__,
            session_id=session_id,
        )

        if raise_on_error:
            raise ChatProviderError(answer) from exc

        return ChatTurn(
            session_id=session_id,
            answer=answer,
            provider=provider,
            model=model,
            status="error",
        )

    def _requested_provider_model(
        self,
        model_options: LLMRequestOptions,
    ) -> tuple[str, str]:
        provider = model_options.provider_mode
        if provider == "auto":
            provider = self.llm_client.provider
        return provider, model_options.model or self.llm_client.model
