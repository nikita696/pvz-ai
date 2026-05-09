from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from pvz_ai.config import Settings, get_settings
from pvz_ai.database import SessionLocal, create_tables
from pvz_ai.gradio_ui import create_gradio_app
from pvz_ai.llm import LLMClient, build_llm_client
from pvz_ai.logging_config import RequestLoggingMiddleware, setup_logging
from pvz_ai.schemas import ChatRequest, ChatResponse, HealthResponse
from pvz_ai.services import ChatService


def create_app(
    settings: Settings | None = None,
    llm_client: LLMClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.auto_create_tables or settings.uses_sqlite_fallback:
            await create_tables(SessionLocal.kw["bind"])
        yield

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestLoggingMiddleware)

    chat_service = ChatService(
        session_factory=SessionLocal,
        llm_client=llm_client or build_llm_client(settings),
        settings=settings,
    )
    app.state.settings = settings
    app.state.chat_service = chat_service

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/chat")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app=settings.app_name,
            environment=settings.app_env,
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        try:
            turn = await chat_service.send_message(
                payload.message,
                session_id=payload.session_id,
                model_options=payload.to_llm_options(),
                raise_on_error=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ChatResponse(
            session_id=turn.session_id,
            answer=turn.answer,
            provider=turn.provider,
            model=turn.model,
            status=turn.status,
            fallback_used=turn.fallback_used,
        )

    gradio_app = create_gradio_app(chat_service)
    gr.mount_gradio_app(app, gradio_app, path="/chat")
    return app


app = create_app()
