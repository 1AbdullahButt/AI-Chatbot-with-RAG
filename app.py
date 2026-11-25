import os
import html
import gradio as gr
from document_parser import parse_file
from rag_engine import (
    build_index, answer_question, DEFAULT_TOP_K, DEFAULT_THRESHOLD,
)

# Every visitor gets their own index stored in gr.State, so people using the
# app at the same time never see each other's documents or conversations.

ALL_DOCS = "All documents"


def file_chips(index):
    # Small HTML list that shows which files are loaded
    if not index:
        return "<div class='empty'>No files loaded yet</div>"
    chips = ""
    for source in index["sources"]:
        count = sum(1 for c in index["chunks"] if c["source"] == source)
        safe_name = html.escape(source)  # stops file names from injecting HTML
        chips += f"<div class='chip'>📎 <b>{safe_name}</b><span>{count} chunks</span></div>"
    return chips


def upload_and_index(files):
    if not files:
        return None, "No files were uploaded.", file_chips(None), gr.Dropdown(choices=[ALL_DOCS], value=ALL_DOCS)

    docs = {}
    messages = []
    for file in files:
        # Gradio can give a path string or a file object, handle both
        path = file if isinstance(file, str) else file.name
        name = os.path.basename(path)
        try:
            pages = parse_file(path)
            pages = [p for p in pages if p["text"].strip()]
            if not pages:
                raise ValueError("no readable text found (is it a scanned file?)")
            docs[name] = pages
            messages.append(f"'{name}' parsed successfully!")
        except Exception as e:
            messages.append(f"Error with '{name}': {e}")

    index = build_index(docs) if docs else None
    choices = [ALL_DOCS] + (index["sources"] if index else [])
    return index, "\n".join(messages), file_chips(index), gr.Dropdown(choices=choices, value=ALL_DOCS)


def clear_files():
    return None, None, "Files cleared.", file_chips(None), gr.Dropdown(choices=[ALL_DOCS], value=ALL_DOCS)


def ask(message, chat, index, top_k, threshold, source):
    chat = chat or []
    answer, diagnostics = answer_question(message, chat, index, int(top_k), threshold, source)
    chat = chat + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", chat, diagnostics


def clear_chat():
    return [], ""


def run_evaluation(tests_text, index, top_k, threshold, source):
    # Each line looks like:  question | keyword1, keyword2
    rows, passed, total = [], 0, 0
    for line in (tests_text or "").splitlines():
        if "|" not in line:
            continue
        question, keywords = line.split("|", 1)
        expected = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        answer, _ = answer_question(question, [], index, int(top_k), threshold, source)
        ok = bool(expected) and all(k in answer.lower() for k in expected)
        total += 1
        passed += ok
        rows.append(f"| {question.strip()} | {', '.join(expected)} | {'✅ pass' if ok else '❌ fail'} |")
    if not total:
        return "Add test lines like: `How many leave days? | 25`"
    table = "| Question | Expected keywords | Result |\n|---|---|---|\n" + "\n".join(rows)
    return f"**Passed {passed} of {total} ({passed / total:.0%})**\n\n{table}"


CSS = """
.gradio-container {width: 100% !important; max-width: 1200px !important; margin: auto;}
.gradio-container .main, .gradio-container .fillable {width: 100% !important; max-width: 1200px !important;}
#hero {text-align: center; padding: 28px 10px 8px 10px;}
#hero h1 {font-size: 2.4rem; margin: 0;
  background: linear-gradient(90deg, #7c3aed, #06b6d4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
#hero p {opacity: 0.75; margin-top: 6px;}
.card {border: 1px solid rgba(124, 58, 237, 0.35); border-radius: 16px;
  padding: 14px; background: rgba(124, 58, 237, 0.06);}
.chip {display: flex; justify-content: space-between; align-items: center;
  padding: 8px 12px; margin: 6px 0; border-radius: 10px;
  background: rgba(6, 182, 212, 0.12); border: 1px solid rgba(6, 182, 212, 0.35);
  font-size: 0.9rem;}
.chip b {text-align: left; flex: 1; margin-left: 6px;}
.chip span {opacity: 0.65; font-size: 0.78rem;}
.empty {opacity: 0.55; padding: 8px 4px; font-size: 0.9rem;}
#steps {font-size: 0.88rem; opacity: 0.8; line-height: 1.7;}
button.primary {background: linear-gradient(90deg, #7c3aed, #06b6d4) !important;
  border: none !important; color: white !important;}
.placeholder, .placeholder-content {display: flex; flex-direction: column;
  align-items: center; justify-content: center; text-align: center; width: 100%;}
.placeholder p {text-align: center; font-size: 1.05rem; opacity: 0.7;}
footer {display: none !important;}
"""

THEME = gr.themes.Soft(primary_hue="violet", secondary_hue="cyan", neutral_hue="slate")

with gr.Blocks(title="Document Q&A Bot") as demo:
    index_state = gr.State(None)  # this visitor's search index

    gr.HTML(
        "<div id='hero'><h1>📄 DocChat AI</h1>"
        "<p>Upload your documents and chat with them. Hybrid search (meaning + keywords) powered by Groq.</p></div>"
    )

    with gr.Row(equal_height=False):
        # Left panel: documents and settings
        with gr.Column(scale=4, elem_classes="card"):
            gr.Markdown("### 📁 Your Documents")
            file_input = gr.File(
                file_count="multiple",
                file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt"],
                label="Drop PDF, Word, PowerPoint, Excel, CSV or TXT files",
            )
            with gr.Row():
                upload_button = gr.Button("⚡ Parse Files", variant="primary")
                clear_files_button = gr.Button("🗑️ Clear")
            upload_output = gr.Textbox(label="Status", interactive=False, lines=2)
            gr.Markdown("#### Loaded files")
            files_panel = gr.HTML(file_chips(None))

            with gr.Accordion("⚙️ Search settings", open=False):
                source_dropdown = gr.Dropdown(choices=[ALL_DOCS], value=ALL_DOCS, label="Search in")
                top_k_slider = gr.Slider(1, 10, value=DEFAULT_TOP_K, step=1, label="Chunks to use (top-k)")
                threshold_slider = gr.Slider(
                    0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.01,
                    label="Similarity threshold (higher = stricter)",
                )

        # Right panel: chat and evaluation
        with gr.Column(scale=7, elem_classes="card"):
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(
                    height=430,
                    placeholder="🔍 I'll browse through your documents to find the answer. Upload files, then ask away.",
                )
                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="Ask anything about your documents...",
                        show_label=False, container=False, scale=8,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)
                new_chat_button = gr.Button("🔄 New conversation")
                with gr.Accordion("📊 Diagnostics (last answer)", open=False):
                    diagnostics = gr.Markdown("Ask a question to see latency, tokens and the chunks used.")

            with gr.Tab("🧪 Evaluation"):
                gr.Markdown(
                    "Check the bot with your own test questions. One per line: "
                    "`question | expected keywords (comma separated)`"
                )
                eval_input = gr.Textbox(lines=5, show_label=False,
                                        placeholder="How many leave days do employees get? | 25")
                eval_button = gr.Button("Run tests", variant="primary")
                eval_output = gr.Markdown()

    settings = [top_k_slider, threshold_slider, source_dropdown]
    upload_button.click(upload_and_index, inputs=[file_input],
                        outputs=[index_state, upload_output, files_panel, source_dropdown])
    clear_files_button.click(clear_files,
                             outputs=[index_state, file_input, upload_output, files_panel, source_dropdown])

    ask_inputs = [question_box, chatbot, index_state] + settings
    question_box.submit(ask, inputs=ask_inputs, outputs=[question_box, chatbot, diagnostics])
    send_button.click(ask, inputs=ask_inputs, outputs=[question_box, chatbot, diagnostics])
    new_chat_button.click(clear_chat, outputs=[chatbot, diagnostics])
    eval_button.click(run_evaluation, inputs=[eval_input, index_state] + settings, outputs=[eval_output])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
