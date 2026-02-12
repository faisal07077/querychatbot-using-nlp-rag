# Query Chatbot using NLP and Retrieval-Augmented Generation (RAG)

## Overview

This project implements a domain-specific chatbot using Natural Language Processing (NLP) and a Retrieval-Augmented Generation (RAG) architecture. The system performs semantic search over a structured knowledge base and generates context-aware responses based on retrieved information.

The chatbot is designed to answer college-related queries using vector embeddings and similarity search.

---

## Key Features

- Semantic search using vector embeddings
- Modular RAG pipeline (retrieval + generation)
- Persistent local vector store
- FAQ-based knowledge ingestion
- Web-based user interface
- Structured and extensible project architecture

---

## Project Architecture

The application follows a modular RAG workflow:

### Data Layer
- `data/faqs.json` — Structured FAQ knowledge base
- `data/vector_store/` — Stored embeddings and index files

### RAG Pipeline (`rag/`)
- `index_builder.py` — Generates embeddings and builds vector index
- `retriever.py` — Performs similarity search over vector store
- `generator.py` — Generates responses using retrieved context
- `rag_engine.py` — Orchestrates retrieval and response generation

### Application Layer
- `app.py` — Flask application entry point
- `templates/chatbot.html` — Frontend interface

---

## Technology Stack

- Python
- Flask
- NLP Embeddings
- Vector Similarity Search
- JSON-based knowledge storage

---

## How It Works

### Indexing Phase (Offline)

1. Load FAQ data from JSON
2. Convert text into embeddings
3. Build a vector index
4. Store embeddings in a local vector database

### Query Phase (Online)

1. User submits a query
2. Query is converted into an embedding
3. Top-K similar documents are retrieved
4. Retrieved context is passed to the generator
5. Context-aware response is returned

---

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/Faisal07077/querychatbot-using-nlp-rag.git
cd querychatbot-using-nlp-rag
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Build the Vector Index 

```bash
python rag/index_builder.py
```

### Run the Application

```bash
python app.py
```

Open your browser and navigate to:

http://127.0.0.1:5000

---

## Folder Structure

(Current structure – may be modified later in future phases)

```
querychatbot-using-nlp-rag/
│
├── app.py
├── nlp.py
├── requirements.txt
├── test_rag.py
│
├── rag/
│   ├── __init__.py
│   ├── generator.py
│   ├── retriever.py
│   ├── index_builder.py
│   └── rag_engine.py
│
├── data/
│   ├── faqs.json
│   └── vector_store/
│
└── templates/
    └── chatbot.html
```

---

## Future Improvements

- Add API key management via environment variables
- Implement re-ranking for improved retrieval accuracy
- Add logging and error handling
- Containerize using Docker
- Deploy on cloud infrastructure
- Add unit and integration testing
