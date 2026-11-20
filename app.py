import os
import re
import gradio as gr
from llm_integration import call_grok_api
from document_parser import parse_file

# Parsed documents are kept in memory: {file name: full text}
parsed_file_content = {}

# Settings for splitting documents into chunks
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks
TOP_CHUNKS = 5         # how many chunks we send to the model

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


def find_best_chunks(question):
    # Score every chunk by how many question keywords it contains
    keywords = get_keywords(question)
    scored = []
    for filename, content in parsed_file_content.items():
        for chunk in split_into_chunks(content):
            chunk_lower = chunk.lower()
            score = sum(chunk_lower.count(k) for k in keywords)
            if score > 0:
                scored.append((score, filename, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:TOP_CHUNKS]


def is_summary_request(question):
    return any(w in question.lower() for w in ["summarize", "summary", "summarise", "summaries"])


def summarize_all_documents():
    summaries = []
    for filename, content in parsed_file_content.items():
        # Only send the first part so very large files do not break the request
        short_content = content[:12000]
        try:
            summary = call_grok_api(f"Summarize the following document:\n\n{short_content}")
            summaries.append(f"**Summary of {filename}:**\n\n{summary}")
        except Exception as e:
            summaries.append(f"Error summarizing {filename}: {e}")
    return "\n\n".join(summaries)


def answer_question(question):
    if not parsed_file_content:
        return "Please upload and parse some files first."

    if is_summary_request(question):
        return summarize_all_documents()

    best = find_best_chunks(question)
    if not best:
        return "I could not find anything relevant in the uploaded files."

    context = "\n\n".join(f"[From {name}]\n{chunk}" for _, name, chunk in best)
    prompt = f"Document excerpts:\n{context}\n\nQuestion: {question}"

    try:
        answer = call_grok_api(prompt)
    except Exception as e:
        return f"Error: {e}"

    sources = sorted({name for _, name, _ in best})
    return f"{answer}\n\n*Sources: {', '.join(sources)}*"


def upload_files(files):
    if not files:
        return "No files were uploaded."

    parsed_file_content.clear()
    messages = []

    for file in files:
        # Gradio can give a path string or a file object, handle both
        path = file if isinstance(file, str) else file.name
        name = os.path.basename(path)
        try:
            parsed_file_content[name] = parse_file(path)
            messages.append(f"'{name}' uploaded and parsed successfully!")
        except Exception as e:
            messages.append(f"Error with '{name}': {e}")

    return "\n".join(messages)


def chat(message, history):
    # history is a list of messages; we only use the latest question
    return answer_question(message)


with gr.Blocks(title="Document Q&A Bot") as demo:
    gr.Markdown("# 📄 Document Q&A Bot\nUpload files, then ask questions about them.")

    with gr.Tab("1. Upload Files"):
        file_input = gr.File(
            file_count="multiple",
            file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".csv"],
            label="Upload Files",
        )
        upload_button = gr.Button("Upload and Parse Files")
        upload_output = gr.Textbox(label="Upload Status")
        upload_button.click(upload_files, inputs=[file_input], outputs=[upload_output])

    with gr.Tab("2. Ask Questions"):
        gr.ChatInterface(
            fn=chat,
            examples=["Summarize the documents", "What are the key points?"],
        )

if __name__ == "__main__":
    demo.launch()
