# Netflix RAG Chatbot System

A full-stack Retrieval-Augmented Generation (RAG) chatbot application that cleans raw dataset files, migrations structural data into a relational SQLite core, generates mathematical coordinates via semantic text embeddings, and hosts a Streamlit conversational AI interface.

## 📊 System Architecture & Pipeline

1. **Data Ingestion & Cleaning (`clean_data.py`)**: Imports the raw dataset, processes missing features, cleans text variations, and standardizes data frames.
2. **Relational Storage Engine (`load_data.py`)**: Structured records are safely parsed, indexed, and stored inside a local **SQLite** database (`netflix.db`).
3. **Mathematical Vectorization Engine (`build_vector_db.py`)**: Queries rows from SQLite, chunks text blocks, and pipes them through the `nomic-embed-text` embedding engine via **Ollama** to convert raw columns into high-dimensional coordinate arrays.
4. **Vector Database Core**: High-dimensional semantic vectors are persisted into a high-performance database cluster (**ChromaDB**) for fast lookup operations.
5. **Search & Evaluation Utility (`query_vector_db.py`)**: Utility logic handling testing, threshold calculations, and semantic comparisons.
6. **Frontend App Engine (`app.py`)**: A user-facing **Streamlit** user interface. It intercepts a user's typed string, converts it to a vector using Nomic, performs a semantic search inside the vector base, constructs a contextual prompt framework, and routes it to the **Gemma 3** LLM server to provide accurate, context-grounded answers.

## 🛠️ Tech Stack & Dependencies

- **LLM Engine:** Ollama (`gemma3`)
- **Embedding Model:** Nomic AI (`nomic-embed-text`)
- **Frontend Framework:** Streamlit
- **Vector Search Database:** ChromaDB / Vector Storage
- **Relational Backend:** SQLite
- **Data Engineering:** Pandas

## 🚀 Local Deployment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd rag-chatbot-project
   ```

2. **Initialize Virtual Environment & Packages:**
   ```bash
   python -m venv .venv
   # Activate environment (Windows)
   .venv\Scripts\activate
   # Activate environment (Mac/Linux)
   source .venv/bin/activate
   
   pip install streamlit pandas chromadb langchain-community ollama
   ```

3. **Initialize the Ollama Mathematical Engine:**
   Ensure Ollama is running on your desktop environment, then fetch the foundational models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull gemma3
   ```

4. **Execute Data Pipeline & Launch Interface:**
   ```bash
   python clean_data.py
   python load_data.py
   python build_vector_db.py
   streamlit run app.py
   ```
