import gradio as gr
from main import process_email, classifier_model, tokenizer, MODEL_FILENAME

def handle_email(email_text):
    if not email_text or not email_text.strip():
        return "<div style='font-weight: bold;'>Please enter email text.</div>", ""
    
    result = process_email(email_text)
    
    priority = str(result.get("priority", "")).lower()
    response_text = result.get("response", "")
    
    if "low" in priority:
        priority_html = """
        <div style='background-color: #28a745; color: white; padding: 12px; border-radius: 8px; font-weight: bold; text-align: center; font-size: 18px;'>
            PRIORITY: LOW
        </div>
        """
    elif "medium" in priority:
        priority_html = """
        <div style='background-color: #ffc107; color: black; padding: 12px; border-radius: 8px; font-weight: bold; text-align: center; font-size: 18px;'>
            PRIORITY: MEDIUM
        </div>
        """
    else:
        priority_html = """
        <div style='background-color: #dc3545; color: white; padding: 12px; border-radius: 8px; font-weight: bold; text-align: center; font-size: 18px;'>
            PRIORITY: HIGH
        </div>
        """
        
    return priority_html, response_text

is_loaded = classifier_model is not None and tokenizer is not None
model_status = "Model Active" if is_loaded else "Model Not Loaded"

with gr.Blocks(title="AI Smart Support Agent") as demo:
    gr.Markdown(
        f"""
        # AI Smart Support
        > **Status:** `{model_status}`
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            email_input = gr.Textbox(
                lines=6, 
                placeholder="Paste email text here...",
                label="Incoming Email"
            )
            
            with gr.Row():
                submit_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.Button("Clear")
            
            
            
        with gr.Column(scale=1):
            priority_output = gr.HTML(label="Priority Level")
            response_output = gr.Textbox(label="Suggested response", lines=8, interactive=False)
            
    submit_btn.click(
        fn=handle_email, 
        inputs=[email_input], 
        outputs=[priority_output, response_output]
    )
    clear_btn.click(
        fn=lambda: ("", "", ""), 
        outputs=[email_input, priority_output, response_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
