import gradio as gr

from pvz_ai.services import ChatService


def create_gradio_app(chat_service: ChatService) -> gr.ChatInterface:
    session_state = gr.State(value=None)

    async def respond(
        message: str,
        history: list[dict[str, str]],
        session_id: str | None,
    ) -> tuple[str, str]:
        del history
        turn = await chat_service.send_message(
            message,
            session_id=session_id,
            raise_on_error=False,
        )
        return turn.answer, turn.session_id

    return gr.ChatInterface(
        fn=respond,
        additional_inputs=[session_state],
        additional_outputs=[session_state],
        title="pvz-ai",
        description="Chat with gpt-oss-120b. Dialog history is saved by session.",
        flagging_mode="never",
        fill_height=True,
        fill_width=True,
        analytics_enabled=False,
        save_history=False,
    )
