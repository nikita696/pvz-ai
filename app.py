import gradio as gr
def respond(message: str) -> str:
    return "You said: " + message
def len(message: str) -> int:
    return len(message)
demo = gr.Interface(fn=respond, inputs="text", outputs="text")
demo.launch()
