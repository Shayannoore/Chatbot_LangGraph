# LangGraph PDF Chatbot 🤖

A powerful, multi-utility chatbot built with **LangGraph** and **Streamlit**. This assistant can engage in intelligent conversations, perform web searches, solve math problems, fetch real-time stock prices, and most importantly, perform **RAG (Retrieval-Augmented Generation)** on uploaded PDF documents.

## 🚀 Features

- **📄 PDF Interaction (RAG)**: Upload any PDF and chat with its content. The bot uses FAISS for vector storage and retrieval.
- **🌐 Web Search**: Seamlessly switches to DuckDuckGo search for general queries outside the PDF's scope.
- **🧮 Calculator**: Built-in tool for basic arithmetic operations.
- **📉 Stock Prices**: Fetch real-time stock data using Alpha Vantage integration.
- **💾 Persistent Memory**: Chat history is saved in a local SQLite database, allowing you to resume past conversations using Thread IDs.
- **📊 Interactive UI**: Clean and modern Streamlit interface with sidebar management for threads and file uploads.

## 🛠️ Tech Stack 

- **Core Framework**: LangGraph
- **LLM**: Google Gemini (via `langchain-google-genai`)
- **Embeddings**: Google Generative AI Embeddings
- **Vector Store**: FAISS
- **Frontend**: Streamlit
- **Persistence**: SQLite (via `SqliteSaver`)
- **Tools**: DuckDuckGo Search, Alpha Vantage API

## 📋 Prerequisites

- Python 3.9+
- A Google AI API Key (for Gemini and Embeddings)
- (Optional) Alpha Vantage API Key for stock price tool

## ⚙️ Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Shayannoore/Chatbot_LangGraph.git
   cd Chatbot_LangGraph
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory and add your credentials:
   ```env
   GOOGLE_API_KEY='your_google_api_key'
   
   # Optional: LangSmith Tracing
   LANGSMITH_TRACING_V2=true
   LANGSMITH_ENDPOINT='https://api.smith.langchain.com'
   LANGSMITH_API_KEY='your_langsmith_api_key'
   LANGSMITH_PROJECT='Chatbot Project'
   ```

## 🏃 Running the App

Start the Streamlit server:
```bash
streamlit run streamlit_frontend.py
```

## 📖 Usage Guide

1. **Upload PDF**: Use the sidebar to upload a document for RAG-based Q&A.
2. **Chat**: Ask questions in the main chat input. The bot will automatically decide whether to use the PDF, web search, or other tools.
3. **Switch Threads**: View and switch between past conversations using the "Past conversations" section in the sidebar.
4. **New Chat**: Click "New Chat" to start a fresh session with a unique Thread ID.

## 📁 Project Structure

- `langgraph_backend.py`: Core logic, state graph definition, and tool implementations.
- `streamlit_frontend.py`: Streamlit UI and interaction logic.
- `requirements.txt`: Project dependencies.
- `chatbot.db`: local SQLite database for conversation persistence.

---
Built with ❤️ using LangGraph and Gemini.
