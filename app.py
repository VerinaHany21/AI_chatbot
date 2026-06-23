import gradio as gr
from engine import process_pdf, route_and_generate


def respond(user_message, history):
    if not user_message.strip():
        return "", history

    reply, label = route_and_generate(user_message, history)

    # Add a visible label so you can see which pipeline was used
    annotated_reply = f"**[{label}]**\n\n{reply}"
    
    # Modern format: Append explicitly as role/content dictionaries
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": annotated_reply})
    return "", history


def upload_pdf(file):
    if file is not None:
        return process_pdf(file.name)
    return "No file uploaded."


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Chatbot with RAG")
    gr.Markdown("Upload a PDF to chat with your documents, or just ask anything directly.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload a PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(
                label="Upload Status",
                value="No PDF uploaded yet.",
                interactive=False
            )
            file_input.upload(fn=upload_pdf, inputs=file_input, outputs=upload_status)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat")
            msg_input = gr.Textbox(
                label="Your message",
                placeholder="Ask something..."
            )
            clear_btn = gr.Button("Clear chat")

            msg_input.submit(fn=respond, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            clear_btn.click(fn=lambda: None, outputs=chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()