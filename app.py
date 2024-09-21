import gradio as gr
import os
import re
from typing import List
from llm_integration import call_grok_api
from document_parser import parse_file

# In-memory storage for parsed content
parsed_file_content = {}

# Function to extract important keywords from the question
def extract_keywords(prompt):
    """Extract important keywords from the user's question"""
    stopwords = ["what", "is", "the", "of", "a", "an", "in", "on", "at", "by", "for"]
    
    # Split the prompt into words and remove stopwords
    keywords = [word for word in re.findall(r'\b\w+\b', prompt.lower()) if word not in stopwords]
    
    return keywords

# Enhanced function to find the most relevant document based on keyword matching
def find_most_relevant_file(prompt):
    """Find the most relevant document based on keyword matching"""
    # Extract keywords from the prompt
    keywords = extract_keywords(prompt)

    best_match = None
    highest_keyword_count = 0
    
    # Search through the documents for keyword matches
    for filename, content in parsed_file_content.items():
        # Count how many keywords appear in the document content
        keyword_count = sum(1 for keyword in keywords if keyword in content.lower())

        # If this document has more matching keywords, it's a better match
        if keyword_count > highest_keyword_count:
            highest_keyword_count = keyword_count
            best_match = filename
    
    return best_match

# Function to check if the user is asking for a summary
def is_summary_request(prompt):
    """Check if the user is asking for a summary"""
    return any(keyword in prompt.lower() for keyword in ["summarize", "summary", "summaries"])

# Gradio function to ask a question
def ask_question(prompt: str):
    """Process a question or a summary request"""
    # Check if the prompt is asking for a summary
    if is_summary_request(prompt):
        return summarize_all_documents()
    
    # Find the most relevant file based on the question
    most_relevant_file = find_most_relevant_file(prompt)
    
    if not most_relevant_file:
        return "No relevant file found."
    
    # Append file content to the prompt
    extended_prompt = f"Here is the content of the file: {parsed_file_content[most_relevant_file]}. Now, {prompt}"
    
    try:
        # Call Groq API with the extended prompt
        response = call_grok_api(extended_prompt)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Function to summarize all uploaded documents
def summarize_all_documents():
    """Summarize each document in the parsed file content with proper formatting"""
    summaries = []
    
    for filename, content in parsed_file_content.items():
        summary_prompt = f"Summarize the following content: {content}"
        try:
            # Call Groq API to summarize the content
            summary = call_grok_api(summary_prompt)
            summaries.append(f"**Summary of {filename}:**\n\n{summary}")
        except Exception as e:
            summaries.append(f"Error summarizing {filename}: {str(e)}")
    
    return "\n\n".join(summaries)

# File upload handler
def upload_files(files: List):
    if files is None or len(files) == 0:
        return "No files were uploaded."

    # Clear old parsed content when new files are uploaded
    parsed_file_content.clear()

    uploaded_files = []

    for file in files:
        # Extract the original file name from the temporary file path
        original_file_name = os.path.basename(file.name)
        
        # Parse the file from the temporary location
        try:
            file_content = parse_file(file.name)

            # Store parsed content in memory
            parsed_file_content[original_file_name.lower()] = file_content
            uploaded_files.append(f"File '{original_file_name}' uploaded and parsed successfully!")
        except Exception as e:
            uploaded_files.append(f"Error uploading '{original_file_name}': {str(e)}")

    return "\n".join(uploaded_files)

# Gradio UI with enhanced design and layout
with gr.Blocks(css=".gradio-container {background-color: #f9f9f9; font-family: Arial;} .gr-box {border-radius: 10px;}") as demo:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1 style="color: #1e88e5;">📄 File Upload & Q&A Bot</h1>
        <p>Upload files and ask questions based on their content</p>
    </div>
    """)
    
    with gr.Tab("Upload Files"):
        with gr.Row():
            file_input = gr.Files(file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".csv"], label="Upload Files", interactive=True)
            upload_button = gr.Button("Upload and Parse Files")
        upload_output = gr.Textbox(label="Upload Status", placeholder="Uploaded files will be displayed here.")
        upload_button.click(upload_files, inputs=[file_input], outputs=[upload_output])

    with gr.Tab("Ask a Question"):
        with gr.Row():
            question_input = gr.Textbox(label="Ask a question based on uploaded files", placeholder="Type your question here...")
            question_button = gr.Button("Submit")
        answer_output = gr.Markdown(label="Answer")
        question_button.click(ask_question, inputs=[question_input], outputs=[answer_output])

# Launch Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
