import logging
import time

import gradio as gr

from pvz_ai.llm import LLMRequestOptions
from pvz_ai.logging_config import emit_structured_log
from pvz_ai.services import SYSTEM_PROMPT, ChatService

logger = logging.getLogger(__name__)


def create_gradio_app(chat_service: ChatService) -> gr.ChatInterface:
    settings = chat_service.settings
    session_state = gr.Textbox(value="", visible=False, label="Session")

    provider_mode = gr.Dropdown(
        choices=[
            ("Groq + Hugging Face fallback", "auto"),
            ("Groq only", "groq"),
            ("Hugging Face only", "huggingface"),
            ("Local echo test", "echo"),
        ],
        value="auto",
        label="Provider",
    )
    model = gr.Textbox(value=settings.llm_model, label="Model")
    temperature = gr.Slider(
        minimum=0,
        maximum=2,
        value=settings.llm_temperature,
        step=0.05,
        label="Temperature",
    )
    top_p = gr.Slider(
        minimum=0,
        maximum=1,
        value=settings.llm_top_p,
        step=0.05,
        label="Top P",
    )
    max_tokens = gr.Slider(
        minimum=128,
        maximum=8192,
        value=settings.llm_max_tokens,
        step=128,
        label="Max tokens",
    )
    system_prompt = gr.Textbox(
        value=SYSTEM_PROMPT,
        label="System prompt",
        lines=3,
        max_lines=6,
    )

    async def respond(
        message: str,
        history: list[dict[str, str]],
        session_id: str | None,
        selected_provider_mode: str,
        selected_model: str,
        selected_temperature: float,
        selected_top_p: float,
        selected_max_tokens: float,
        selected_system_prompt: str,
    ) -> tuple[str, str]:
        start = time.perf_counter()
        history_length = len(history or [])
        session_id = session_id or None
        options = LLMRequestOptions(
            provider_mode=selected_provider_mode or "auto",
            model=(selected_model or "").strip() or None,
            temperature=float(selected_temperature),
            top_p=float(selected_top_p),
            max_tokens=int(selected_max_tokens),
            system_prompt=(selected_system_prompt or "").strip() or None,
        )

        emit_structured_log(
            logger,
            logging.INFO,
            "gradio_chat_submitted",
            provider_mode=options.provider_mode,
            model=options.model or settings.llm_model,
            message_chars=len(message or ""),
            history_items=history_length,
            session_id=session_id or "-",
        )

        try:
            turn = await chat_service.send_message(
                message,
                session_id=session_id,
                model_options=options,
                raise_on_error=False,
            )
        except Exception:
            logger.exception(
                "gradio chat failed provider_mode=%s model=%s",
                options.provider_mode,
                options.model or settings.llm_model,
                extra={"session_id": session_id or "-"},
            )
            emit_structured_log(
                logger,
                logging.ERROR,
                "gradio_chat_failed",
                provider_mode=options.provider_mode,
                model=options.model or settings.llm_model,
                session_id=session_id or "-",
            )
            raise

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        emit_structured_log(
            logger,
            logging.INFO,
            "gradio_chat_completed",
            provider=turn.provider,
            model=turn.model,
            status=turn.status,
            fallback_used=turn.fallback_used,
            elapsed_ms=elapsed_ms,
            session_id=turn.session_id,
        )
        return turn.answer, turn.session_id

    return gr.ChatInterface(
        fn=respond,
        additional_inputs=[
            session_state,
            provider_mode,
            model,
            temperature,
            top_p,
            max_tokens,
            system_prompt,
        ],
        additional_outputs=[session_state],
        additional_inputs_accordion=gr.Accordion(
            label="Model settings",
            open=False,
        ),
        title="pvz-ai",
        textbox=gr.Textbox(
            placeholder="Напиши сообщение",
            show_label=False,
            container=False,
        ),
        flagging_mode="never",
        fill_height=True,
        fill_width=True,
        analytics_enabled=False,
        save_history=False,
    )
