# AI-Powered Health Assistant v0.3

## Phase 3 & 4 – Memory + Retrieval-Augmented Generation (RAG)

An intelligent AI health assistant powered by **Gemini 2.5 Flash**, **LangChain**, **ChromaDB**, and **Conversational Memory**.

### What's New

- **Conversational Memory** – The assistant remembers previous messages within a session (name, age, conditions, etc.)
- **RAG Pipeline** – Load medical PDFs, split into chunks, embed with Gemini, store in ChromaDB, and retrieve relevant context for grounded answers
- **Modular Architecture** – Clean separation into `memory/`, `rag/`, and `prompts/` packages

---

## Folder Structure

```
HealthAssistant/
├── app.py                    # Main application entry point
├── config.py                 # Configuration & environment variables
├── pyproject.toml            # Dependencies (managed by uv)
│
├── prompts/
│   ├── __init__.py
│   ├── system_prompt.py      # System prompt with memory + RAG placeholders
│   └── templates.py          # ChatPromptTemplate definition
│
├── memory/
│   ├── __init__.py
│   ├── memory_manager.py     # Chat history management
│   └── session.py            # User session/profile data
│
├── rag/
│   ├── __init__.py
│   ├── loader.py             # PDF document loader
│   ├── splitter.py           # Text chunking
│   ├── embeddings.py         # Gemini embedding model
│   ├── vectorstore.py        # ChromaDB operations
│   ├── retriever.py          # Similarity search retriever
│   └── pipeline.py           # End-to-end RAG orchestrator
│
├── data/
│   └── medical_docs/         # Place medical PDFs here
│
└── .env.example              # Example environment variables template
```

---

## Security

- Do not commit `.env`, `atlas-credentials.env`, or any secret credentials.
- Create a local `.env` file from `.env.example` and keep it out of version control.
- `.venv/`, `Medical Documents/`, and `data/chroma_db/` are now ignored to prevent accidental commits of local environments and large data.
- Rotate any exposed API keys or credentials immediately.

---

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure API Key

Create/edit `.env` in the `HealthAssistant/` directory:

```
GOOGLE_API_KEY=your_api_key_here
```

### 3. (Optional) Add Medical Documents

Place PDF files in `data/medical_docs/`. The RAG pipeline will automatically load, chunk, embed, and index them on startup.

### 4. Run

```bash
uv run python app.py
```

---

## Commands

| Command     | Description                            |
|-------------|----------------------------------------|
| `exit/quit` | Exit the assistant                     |
| `clear`     | Clear conversation memory              |
| `rebuild`   | Rebuild the document vector index      |

---

## Technologies

| Component        | Technology                          |
|------------------|-------------------------------------|
| Framework        | LangChain                           |
| LLM              | Gemini 2.5 Flash                    |
| Memory           | LangChain ChatMessageHistory        |
| Embeddings       | Google Generative AI Embeddings     |
| Vector Database  | ChromaDB                            |
| Document Loader  | PyPDFLoader                         |
| Text Splitter    | RecursiveCharacterTextSplitter      |
| Package Manager  | uv                                  |

---

## Architecture

```
User Question
      │
      ├──────────────────┐
      ▼                  ▼
Conversation Memory    RAG Retriever
      │               (Medical PDFs)
      ▼                  │
Chat History      Relevant Chunks
      └────────┬─────────┘
               ▼
      ChatPromptTemplate
               ▼
          Gemini API
               ▼
        StrOutputParser
               ▼
          Final Answer
```
