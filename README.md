# Project Report: AI-Driven Document Q&A Bot using Gradio and Groq API

## 1. Project Overview
The project aims to build an AI-driven chatbot capable of answering questions based on the content of uploaded documents. The bot uses advanced language models (Groq API) to process various file types, extract relevant information, and generate accurate answers. Users can upload files in different formats (PDFs, Word documents, Excel sheets, PowerPoint slides, and CSV files), and the bot allows them to interactively ask questions or get summaries of the uploaded content.

## 2. Objectives
- Develop a tool to upload and parse multiple document types.
- Implement a Q&A system capable of answering questions based on the content of the uploaded documents.
- Integrate with the Groq API for advanced language processing and retrieval-augmented generation (RAG).
- Provide a user-friendly interface using Gradio to enhance interaction with the bot.

## 3. Methodology

### 3.1 Project Structure & Modules
The project was developed using three main Python modules:
1. **App.py**: The main module handling the Gradio interface and interaction between the user and the chatbot.
2. **DocumentParser.py**: Responsible for reading and extracting content from different document formats.
3. **LLMIntegration.py**: Interacts with the Groq API to generate answers and summaries from the parsed document content.

### 3.2 Document Parsing Process
The `DocumentParser.py` module reads various document formats using specialized libraries:
- **PDF files**: Extracted using the PyMuPDF library (`fitz`).
- **DOCX files**: Parsed using `python-docx`.
- **PPTX files**: Processed using `python-pptx`.
- **XLSX files**: Handled using `openpyxl`.
- **CSV files**: Read using Python's built-in `csv` module.

### 3.3 Retrieval-Augmented Generation (RAG) Process
The project uses RAG, where:
- **Retrieval**: Searches and identifies the relevant content within the uploaded documents based on the user's question.
- **Generation**: Uses the Groq API to generate responses from the retrieved content.

### 3.4 Gradio UI Integration
- **Gradio** was used to create a user-friendly web-based interface.
- The interface allows:
  - **File Upload**: Users can upload multiple files, which are parsed and stored in memory.
  - **Question Asking**: Users ask questions, and the chatbot retrieves answers from the relevant document content.

