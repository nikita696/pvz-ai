import json
import time
from collections.abc import AsyncIterator
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from pvz_ai.config import Settings
from pvz_ai.llm import LLMRequestOptions, ProviderMode
from pvz_ai.schemas import OpenAIChatCompletionRequest
from pvz_ai.services import ChatProviderError, ChatService


def create_openai_router(chat_service: ChatService, settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models() -> dict:
        models = [
            _model_card(settings.llm_model, "pvz-ai"),
            _model_card(f"{settings.llm_model}:groq", "groq"),
            _model_card(f"{settings.llm_model}:huggingface", "huggingface"),
        ]
        if settings.llm_provider == "echo":
            models.append(_model_card("echo-local", "pvz-ai"))
        return {"object": "list", "data": models}

    @router.post("/v1/chat/completions")
    async def chat_completions(
        payload: OpenAIChatCompletionRequest,
        request: Request,
    ):
        provider_mode, provider_model, response_model = _resolve_model(
            requested_model=payload.model,
            default_model=settings.llm_model,
        )
        max_tokens = payload.max_tokens or payload.max_completion_tokens
        options = LLMRequestOptions(
            provider_mode=provider_mode,
            model=provider_model,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=max_tokens,
        )
        session_id = _session_id_from_request(payload, request)

        try:
            turn = await chat_service.complete_messages(
                payload.normalized_messages(),
                session_id=session_id,
                model_options=options,
                raise_on_error=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ChatProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        completion_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())
        if payload.stream:
            return StreamingResponse(
                _stream_completion_chunks(
                    completion_id=completion_id,
                    created=created,
                    model=response_model,
                    content=turn.answer,
                ),
                media_type="text/event-stream",
            )
        return _completion_response(
            completion_id=completion_id,
            created=created,
            model=response_model,
            content=turn.answer,
            provider=turn.provider,
            fallback_used=turn.fallback_used,
            session_id=turn.session_id,
        )

    return router


def _model_card(model_id: str, owned_by: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": owned_by,
    }


def _resolve_model(
    requested_model: str | None,
    default_model: str,
) -> tuple[ProviderMode, str, str]:
    model = requested_model or default_model
    if model == "echo-local":
        return "echo", "echo-local", "echo-local"
    for suffix, provider in (
        (":groq", "groq"),
        (":huggingface", "huggingface"),
    ):
        if model.endswith(suffix):
            return provider, model.removesuffix(suffix), model
    return "auto", model, model


def _session_id_from_request(
    payload: OpenAIChatCompletionRequest,
    request: Request,
) -> str | None:
    metadata = payload.metadata or {}
    raw_session_id = (
        _string_or_none(metadata.get("session_id"))
        or _string_or_none(metadata.get("chat_id"))
        or request.headers.get("x-openwebui-chat-id")
        or request.headers.get("x-openwebui-user-id")
        or payload.user
    )
    if not raw_session_id:
        return None
    return str(uuid5(NAMESPACE_URL, f"pvz-ai:openwebui:{raw_session_id}"))


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _completion_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
    provider: str,
    fallback_used: bool,
    session_id: str,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "pvz_ai": {
            "provider": provider,
            "fallback_used": fallback_used,
            "session_id": session_id,
        },
    }


async def _stream_completion_chunks(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
) -> AsyncIterator[str]:
    yield _sse_chunk(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
    )
    yield _sse_chunk(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
        }
    )
    yield _sse_chunk(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


def _sse_chunk(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
