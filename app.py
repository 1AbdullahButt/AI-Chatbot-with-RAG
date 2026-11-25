import os
import re
import html
import gradio as gr
from llm_integration import call_grok_api
from document_parser import parse_file

# Each visitor gets their own documents ({file name: full text}) stored in
# gr.State, so people using the app at the same time never see each other's files.

# Settings for splitting documents into chunks
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks
TOP_CHUNKS = 5         # how many chunks we send to the model
SUMMARY_LIMIT = 12000  # max characters of one file we summarize

STOPWORDS = {
    "what", "is", "the", "of", "a", "an", "in", "on", "at", "by", "for",
    "and", "to", "are", "was", "how", "who", "when", "where", "why", "me",
    "tell", "about", "does", "do", "this", "that", "with", "from",
}


def get_keywords(text):
    # Lowercase words without stopwords
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def split_into_chunks(text):
    # Cut the text into overlapping pieces
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def find_best_chunks(question, docs):
    # Score every chunk by how many question keywords it contains
    keywords = get_keywords(question)
    scored = []
    for filename, content in docs.items():
        for chunk in split_into_chunks(content):
            chunk_lower = chunk.lower()
            score = sum(chunk_lower.count(k) for k in keywords)
            if score > 0:
                scored.append((score, filename, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:TOP_CHUNKS]


def first_chunks(docs):
    # Used for general questions like "what is this about?"
    # We take the beginning of each document
    result = []
    for filename, content in docs.items():
        result.append((0, filename, content[:CHUNK_SIZE * 2]))
    return result[:TOP_CHUNKS]


def is_summary_request(question):
    # Only short questions count, so "does the summary table show Q3?" is a normal question
    words = question.lower().split()
    wants_summary = any(w.strip("?.!,") in ["summarize", "summarise", "summary", "summaries"] for w in words)
    return wants_summary and len(words) <= 8


def summarize_all_documents(docs):
    summaries = []
    for filename, content in docs.items():
        short_content = content[:SUMMARY_LIMIT]
        try:
            summary = call_grok_api(f"Summarize the following document:\n\n{short_content}")
            note = ""
            if len(content) > SUMMARY_LIMIT:
                note = f"\n\n*(Based on the first {SUMMARY_LIMIT:,} characters of this file.)*"
            summaries.append(f"**Summary of {filename}:**\n\n{summary}{note}")
        except Exception as e:
            summaries.append(f"Error summarizing {filename}: {e}")
    return "\n\n".join(summaries)


def answer_question(question, docs):
    if not docs:
        return "Please upload and parse some files first (left panel)."

    if is_summary_request(question):
        return summarize_all_documents(docs)

    best = find_best_chunks(question, docs)
    if not best:
        # No keyword matched, so use the start of the documents
        best = first_chunks(docs)

    context = "\n\n".join(f"[From {name}]\n{chunk}" for _, name, chunk in best)
    prompt = f"Document excerpts:\n{context}\n\nQuestion: {question}"

    try:
        answer = call_grok_api(prompt)
    except Exception as e:
        return f"Error: {e}"

    sources = sorted({name for _, name, _ in best})
    return f"{answer}\n\n*Sources: {', '.join(sources)}*"


def upload_files(files):
    docs = {}
    if not files:
        return docs, "No files were uploaded."

    messages = []
    for file in files:
        # Gradio can give a path string or a file object, handle both
        path = file if isinstance(file, str) else file.name
        name = os.path.basename(path)
        try:
            text = parse_file(path)
            if not text.strip():
                raise ValueError("no readable text found (is it a scanned file?)")
            docs[name] = text
            messages.append(f"'{name}' uploaded and parsed successfully!")
        except Exception as e:
            messages.append(f"Error with '{name}': {e}")

    return docs, "\n".join(messages)


def file_chips(docs):
    # Small HTML list that shows which files are loaded
    if not docs:
        return "<div class='empty'>No files loaded yet</div>"
    chips = ""
    for name, text in docs.items():
        safe_name = html.escape(name)  # stops file names from injecting HTML
        chips += f"<div class='chip'>📎 <b>{safe_name}</b><span>{len(text):,} chars</span></div>"
    return chips


def upload_and_show(files):
    docs, status = upload_files(files)
    return docs, status, file_chips(docs)


def clear_files():
    return {}, None, "Files cleared.", file_chips({})


def chat(message, history, docs):
    # The text box can send a dict like {"text": "...", "files": []}
    if isinstance(message, dict):
        message = message.get("text", "")
    return answer_question(message, docs or {})


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
    docs_state = gr.State({})  # this visitor's documents
    gr.HTML(
        "<div id='hero'><h1>📄 DocChat AI</h1>"
        "<p>Upload your documents and chat with them. Powered by RAG + Groq.</p></div>"
    )

    with gr.Row(equal_height=False):
        # Left panel: upload
        with gr.Column(scale=4, elem_classes="card"):
            gr.Markdown("### 📁 Your Documents")
            file_input = gr.File(
                file_count="multiple",
                file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".csv"],
                label="Drop PDF, Word, PowerPoint, Excel or CSV files",
            )
            with gr.Row():
                upload_button = gr.Button("⚡ Parse Files", variant="primary")
                clear_button = gr.Button("🗑️ Clear")
            upload_output = gr.Textbox(label="Status", interactive=False, lines=2)
            gr.Markdown("#### Loaded files")
            files_panel = gr.HTML(file_chips({}))
            gr.Markdown(
                "**How it works**<br>1. Upload files<br>2. Click Parse Files<br>"
                "3. Ask anything on the right",
                elem_id="steps",
            )

        # Right panel: chat
        with gr.Column(scale=7, elem_classes="card"):
            gr.Markdown("### 💬 Ask Questions")
            gr.ChatInterface(
                fn=chat,
                additional_inputs=[docs_state],
                textbox=gr.MultimodalTextbox(placeholder="Ask anything about your documents...", file_count="none", sources=[]),
                chatbot=gr.Chatbot(height=460, placeholder="🔍 I'll browse through your documents to find the answer. Upload files, then ask away."),
            )

    upload_button.click(upload_and_show, inputs=[file_input], outputs=[docs_state, upload_output, files_panel])
    clear_button.click(clear_files, outputs=[docs_state, file_input, upload_output, files_panel])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
