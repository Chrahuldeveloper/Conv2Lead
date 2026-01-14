# Knowledge-Based AI Agent (RAG System)

This project implements a **knowledge-grounded AI agent** that answers user questions using a **local knowledge base** via **Retrieval-Augmented Generation (RAG)**.

The agent retrieves relevant context from stored documents and generates answers strictly based on that context using an LLM.

---

## ✨ Key Features

- Local knowledge base ingestion (Markdown)
- Chunking & embeddings pipeline
- Vector search using ChromaDB
- Agent reasoning & intent detection
- Context-grounded LLM responses
- Clean modular architecture
- Hot-reload support for development
---

## 🧠 Agent Capabilities

The agent:
- Detects user intent
- Retrieves relevant knowledge using vector similarity
- Generates answers **only from retrieved context**
- Responds with *“I don’t know”* when information is missing

---

## 📁 Project Structure

```

├── agent/
│   ├── agent.py          
│   └── state.py         
│
├── rag/
│   └── vectorstore.py
|   └── generator.py       
│
├── data/
│   └── knowledge.md      
│
├── main.py               
├── config.py               
├── requirements.txt
└── README.md

```

---

## 📚 Knowledge Base

Stored locally in `data/knowledge.md`.

Example content:
- Pricing & feature plans
- Company policies
- Support details

The file is automatically:
1. Chunked
2. Embedded
3. Stored in a vector database

---

## 🔄 RAG Pipeline

1. Load `knowledge.md`
2. Split into semantic chunks
3. Generate embeddings (Sentence Transformers)
4. Store in ChromaDB
5. Retrieve top-K relevant chunks
6. Inject context into LLM prompt
7. Generate grounded answer

---

## 🤖 Agent Workflow

```

User Question
↓
Intent Detection
↓
Vector Retrieval (RAG)
↓
Context Assembly
↓
LLM Generation
↓
Final Answer

````

---

## 🧩 Task Requirements Mapping

### 1. Agent reasoning & intent detection
- Implemented via `AgentState`
- Routes queries through correct execution path

### 2. Correct use of RAG
- Vector search performed **before** LLM call
- LLM answers strictly from retrieved context

### 3. Clean state management
- `AgentState` object tracks question, intent, context, answer

### 4. Proper tool calling logic
- Retrieval and generation are clearly separated
- No direct LLM calls without context

### 5. Code clarity & structure
- Modular folder-based architecture
- Single responsibility per file

### 6. Real-world deployability
- Local vector DB persistence
- Hot-reload support
- Easily extensible to API

---

## 🛠 Tech Stack

- **Python 3.10+**
- **SentenceTransformers** – embeddings
- **ChromaDB** – vector database
- **LangChain Text Splitters**
- **Gemini** – answer generation

---

## ▶️ Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Start the agent (with hot reload)

```bash
watchfiles "python main.py"
```

---

## 🧪 Example Query

```text
User: What are the plans available?
Agent:
- Basic Plan: $29/month, 10 videos, 720p
- Pro Plan: $79/month, unlimited videos, 4K, AI captions
```

---

## 🚀 Future Improvements

* Streaming responses
* Conversation memory
* REST API / FastAPI interface
* UI dashboard
* Multi-document support

---

## 📝 Notes

* The agent **does not hallucinate**
* Answers are strictly grounded in local knowledge
* Safe fallback behavior for unknown queries

---

## 📌 Summary

This project demonstrates a **production-ready AI agent** using best practices in:

* RAG
* Agent architecture
* Vector databases
* Clean state management

Designed to be **simple, reliable, and extensible**.


