# 📄 ChatPDF with RAG System

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using LangChain, OpenAI, and FAISS. Built with Python and Streamlit.

![Demo]

## 🚀 Features

- **Document Processing**: Extract text from PDFs with page-aware chunking
- **Vector Search**: FAISS-based semantic search with OpenAI embeddings
- **Question Answering**: GPT-3.5-turbo powered responses with source citations
- **Secure API Handling**: Environment-based key management
- **Modern UI**: Streamlit interface with dark theme and responsive design

## ⚙️ Architecture

```mermaid
graph TD
    A[PDF Upload] --> B[Text Extraction]
    B --> C[Chunking]
    C --> D[Embedding Generation]
    D --> E[FAISS Vector Store]
    E --> F[Query Processing]
    F --> G[LLM Response Generation]