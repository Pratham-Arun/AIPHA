# 🩺 AI-Powered Health Assistant

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/LangChain-0.1.0%2B-green.svg)](https://python.langchain.com/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-orange.svg)](https://deepmind.google/technologies/gemini/)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB%20%26%20GridFS-cyan.svg)](https://www.mongodb.com/)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-blueviolet.svg)](https://www.trychroma.com/)
[![Package Manager](https://img.shields.io/badge/Package%20Manager-uv-pink.svg)](https://github.com/astral-sh/uv)

An intelligent, enterprise-grade AI Health Assistant built on a modular Retrieval-Augmented Generation (RAG) architecture. It connects **Gemini 2.5 Flash**, **LangChain**, a persistent **MongoDB & GridFS** binary storage backend, and a local **ChromaDB** vector database. The system is designed to provide highly grounded, context-aware health education while prioritizing local session memory, duplicate document checking, and robust file management.

---

## 📌 Table of Contents

- [🩺 AI-Powered Health Assistant](#-ai-powered-health-assistant)
  - [📌 Table of Contents](#-table-of-contents)
  - [🔍 Project Overview](#-project-overview)
    - [Problem Statement](#problem-statement)
    - [Objectives](#objectives)
  - [✨ Key Features](#-key-features)
  - [🏗️ System Architecture](#️-system-architecture)
    - [End-to-End Workflow](#end-to-end-workflow)
    - [MongoDB → GridFS → ChromaDB Architecture](#mongodb--gridfs--chromadb-architecture)
    - [Conversation Memory Architecture](#conversation-memory-architecture)
    - [RAG Pipeline Details](#rag-pipeline-details)
  - [🛠️ Technology Stack](#️-technology-stack)
  - [📂 Project Structure](#-project-structure)
  - [🚀 Installation \& Setup](#-installation--setup)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Clone and Sync Dependencies](#2-clone-and-sync-dependencies)
    - [3. Migrate Local Files to Database (Optional)](#3-migrate-local-files-to-database-optional)
  - [⚙️ Environment Variables](#️-environment-variables)
  - [🏃 Running the Project](#-running-the-project)
  - [💻 CLI Commands](#-cli-commands)
  - [📅 Detailed Phase-by-Phase Implementation](#-detailed-phase-by-phase-implementation)
    - [Phase 1: Environment \& Base Configuration](#phase-1-environment--base-configuration)
    - [Phase 2: MongoDB \& GridFS Document Store](#phase-2-mongodb--gridfs-document-store)
    - [Phase 3: Conversational Memory \& User Profiling](#phase-3-conversational-memory--user-profiling)
    - [Phase 4: Vector Store \& RAG Pipeline Setup](#phase-4-vector-store--rag-pipeline-setup)
    - [Phase 5: Incremental Indexing \& System Orchestration](#phase-5-incremental-indexing--system-orchestration)
  - [📈 Current Project Status](#-current-project-status)
  - [🗺️ Future Roadmap (Phases 6–10)](#️-future-roadmap-phases-610)
  - [📚 References](#-references)
  - [📄 License](#-license)

---

## 🔍 Project Overview

### Problem Statement
Standard Large Language Models (LLMs) often suffer from medical hallucinations or lack access to domain-specific, authoritative health literature (such as official WHO guidelines or peer-reviewed research). Furthermore, generic chatbots lack session-specific context, forgetting critical user details like allergies, current medications, or age during a continuous conversation.

### Objectives
1. **Fact-Grounded Answers:** Ground all model answers in trusted local medical literature using semantic search, fallback to general medical knowledge only when documentation is absent, and state the source.
2. **Persistent Document Backend:** Store binary medical PDFs securely in MongoDB GridFS, keeping metadata and index states synchronized.
3. **Optimized Local Embedding Index:** Convert PDF documents into semantic chunks via localized HuggingFace sentence-transformers and index them incrementally inside ChromaDB.
4. **Session-Aware Interaction:** Maintain a clean conversational memory and user profile context (e.g. dietary restrictions, age, weight) across conversation turns.

---

## ✨ Key Features

*   **Conversational Memory Manager:** Tracks message history using LangChain's `ChatMessageHistory` and dynamically builds and updates user profiles (name, age, weight, height, medical conditions, allergies, and dietary preferences).
*   **GridFS Document Store:** Stores original binary PDFs directly in MongoDB and tracks metadata (indexed status, page counts, chunk counts, source organization, size, and hash) in a standard `documents` collection.
*   **Incremental Indexing & Deduplication:** Avoids wasteful re-embedding of already processed files on startup. Calculates SHA256 hashes of PDF files to skip indexing duplicate uploads.
*   **ChromaDB Vector Store & Retriever:** Embeds chunks locally using the HuggingFace `all-MiniLM-L6-v2` model. Uses distance scores to filter out low-confidence context.
*   **Command Line Workspace Control:** Offers interactive CLI commands to upload documents, list files, view system health, clear memory, and force database rebuilds.
*   **Windows Handle-Lock Mitigation:** Cleanly releases SQLite/ChromaDB file-handles and triggers Python garbage collection before running index rebuilds, preventing `WinError 32` (Permission Denied).
*   **Custom Source Metadata Filtering:** Restricts searches dynamically (e.g., to only "WHO", "CDC", or "NIH" documents) when specific organizations are mentioned in user queries.

---

## 🏗️ System Architecture

### End-to-End Workflow

```
       +-------------------+             +-----------------------+
       |   User Message    |             |  Medical Documents    |
       +---------+---------+             +-----------+-----------+
                 |                                   |
                 v                                   v
       +---------+---------+             +-----------+-----------+
       |   Memory Manager  |             |  MongoDB GridFS Store |
       | (Session Profile) |             +-----------+-----------+
       +---------+---------+                         | (Incremental)
                 |                                   v
                 |                       +-----------+-----------+
                 |                       |  ChromaDB Vector Store|
                 |                       +-----------+-----------+
                 |                                   |
                 +-----------------+-----------------+
                                   | (Context + History)
                                   v
                       +-----------+-----------+
                       |   ChatPromptTemplate  |
                       +-----------+-----------+
                                   |
                                   v
                       +-----------+-----------+
                       |    Gemini 2.5 LLM     |
                       +-----------+-----------+
                                   |
                                   v
                       +-----------+-----------+
                       |   Grounded Response   |
                       +-----------------------+
```

### MongoDB → GridFS → ChromaDB Architecture

```
                 [Local PDF / CLI Upload]
                            │
                            ▼
           ┌─────────────────────────────────┐
           │        DocumentService          │
           └──────┬───────────────────┬──────┘
                  │                   │
                  ▼                   ▼
           ┌──────────────┐   ┌──────────────┐
           │ MongoDB      │   │ GridFS       │
           │ metadata     │   │ (fs.files &  │
           │ collection   │   │  fs.chunks)  │
           └──────┬───────┘   └──────┬───────┘
                  │                  │
                  │ (Check Pending)  │ (Temp Download)
                  ▼                  ▼
           ┌─────────────────────────────────┐
           │       Incremental Indexer       │
           │   - Compute SHA256 Hash         │
           │   - Split PDF into Text Chunks   │
           │   - Generate HF Embeddings      │
           └──────────────────┬──────────────┘
                              │
                              ▼
           ┌─────────────────────────────────┐
           │      ChromaDB Vector Store      │
           │    (collection: medical_docs)   │
           └─────────────────────────────────┘
```

### Conversation Memory Architecture

```
                  ┌─────────────────────────────────────┐
                  │            MemoryManager            │
                  └──────┬───────────────────────┬──────┘
                         │                       │
                         ▼                       ▼
           ┌───────────────────────────┐   ┌───────────────────────────┐
           │          Session          │   │    ChatMessageHistory     │
           │ (Volatile User Profile)   │   │     (LangChain Track)     │
           │ ├─ Name                   │   │ ├─ [HumanMessage,         │
           │ ├─ Age                    │   │ │   AIMessage, ...]       │
           │ ├─ Weight & Height        │   │ ├─ Keeps last 20 messages │
           │ ├─ Medical Conditions     │   │ └─ Clears with 'clear'    │
           │ └─ Allergies & Diets      │   └───────────────────────────┘
           └───────────────────────────┘
```

### RAG Pipeline Details
1. **Query Inspection:** Scans user queries for key organization indicators (e.g., "WHO", "CDC", "NIH") and attaches a metadata query filter (`source: { $contains: "ORG" }`) to ChromaDB search.
2. **Similarity Search:** Queries ChromaDB for `RETRIEVER_TOP_K` chunks using the local embedding model.
3. **Relevance Thresholding:** Filters chunks using a distance score threshold. Any chunk with a distance greater than `RETRIEVER_DISTANCE_THRESHOLD` is discarded.
4. **Answer Grounding:** Prompt instructs Gemini to state if facts are missing from documents and clearly cite source filenames and pages when context is utilized.

---

## 🛠️ Technology Stack

| Component | Technology | Why Chosen? |
| :--- | :--- | :--- |
| **LLM** | Gemini 2.5 Flash | High execution speed, large context window, excellent structured prompt adherence, and cost efficiency. |
| **Framework** | LangChain | Standardizes vector database loading, prompt chaining, session messaging, and retrievers. |
| **Metadata DB** | MongoDB | Rich querying features, dynamic schema-less architecture, perfect for variable document meta. |
| **File Storage** | GridFS | Built into MongoDB; allows storing large PDFs (exceeding BSON 16MB limit) directly alongside metadata records. |
| **Vector DB** | ChromaDB | Highly portable, local file persistence, fast installation, and clean compatibility with LangChain. |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Completely offline model, generates accurate sentence-level embeddings without incurring API costs. |
| **Package Tool**| uv | Blazing fast replacement for `pip`, manages virtual environments, lockfiles, and dependencies instantly. |

---

## 📂 Project Structure

```
HealthAssistant/
├── .env.example                # Template for credentials
├── app.py                      # Interactive CLI client and execution loop
├── audit_pipeline.py           # Integrity auditor checking MongoDB vs ChromaDB
├── config.py                   # Configuration and path validation logic
├── pyproject.toml              # Project dependencies managed by uv
├── uv.lock                     # UV dependency lockfile
├── LICENSE                     # MIT License file
├── README.md                   # System documentation
│
├── database/                   # Database orchestration package
│   ├── __init__.py
│   ├── connection.py           # Singleton manager for MongoDB client
│   ├── document_service.py     # High-level facade for uploader, metadata, & GridFS
│   ├── gridfs_manager.py       # MongoDB GridFS read/write functions
│   ├── metadata.py             # MongoDB metadata CRUD collection operations
│   ├── migrate.py              # Script to migrate local files to GridFS
│   └── uploader.py             # Core PDF uploading and verification engine
│
├── memory/                     # Conversation history package
│   ├── __init__.py
│   ├── memory_manager.py       # Wrapper for LangChain ChatMessageHistory
│   └── session.py              # User profiles and health metric maps
│
├── prompts/                    # AI Prompt templating package
│   ├── __init__.py
│   ├── system_prompt.py        # System prompt with RAG & memory directives
│   └── templates.py            # LangChain ChatPromptTemplate definitions
│
├── rag/                        # Retrieval-Augmented Generation package
│   ├── __init__.py
│   ├── embeddings.py           # SentenceTransformers embedding loader
│   ├── indexer.py              # Incremental index orchestrator and hash checker
│   ├── loader.py               # PyPDF-based local document reader
│   ├── pipeline.py             # End-to-end index builder and search retriever
│   ├── retriever.py            # ChromaDB query processor with scoring
│   └── splitter.py             # Text chunking rules (RecursiveCharacterTextSplitter)
│
└── data/                       # Local data directories (gitignored)
    ├── chroma_db/              # Persisted ChromaDB SQLite vector files
    └── temp/                   # Temporary directory for indexer PDF downloads
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.11 or higher
- MongoDB instance running locally (e.g. `mongodb://localhost:27017`) or a remote MongoDB Atlas URI.
- [uv Package Manager](https://github.com/astral-sh/uv) (Highly Recommended):
  ```bash
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. Clone and Sync Dependencies
Navigate to the `HealthAssistant` directory and run:
```bash
uv sync
```

### 3. Migrate Local Files to Database (Optional)
If you have existing PDFs inside a local directory and want to seed your MongoDB instance, run:
```bash
uv run python database/migrate.py
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root by copying the provided template:

```bash
copy .env.example .env
```

Then update `.env` with your credentials and MongoDB connection details.

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | Your Google Gemini API Key | *Required* |
| `MONGO_URI` | Connection string for MongoDB server | `mongodb://localhost:27017` |
| `MONGO_DB_NAME`| Name of the target database | `health_assistant` |

Example `.env` contents:

```env
GOOGLE_API_KEY=your-google-api-key
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=health_assistant
```

---

## 🏃 Running the Project

To launch the interactive CLI chatbot:
```bash
uv run python app.py
```

To run the pipeline verification diagnostic script:
```bash
uv run python audit_pipeline.py
```

To execute the manual verification tests:
```bash
uv run python test_rag.py
```

---

## 💻 CLI Commands

When running `app.py`, you can enter standard health questions or use these special system commands:

| Command | Action |
| :--- | :--- |
| `upload` | Triggers a file selector prompt. Input the path to a local PDF, and supply the document title, category, and source organization. |
| `docs` | Lists all documents stored in MongoDB along with their indexing status and organizations. |
| `rebuild` | Forces a complete rebuild of the vector index (resets MongoDB indexed states, wipes ChromaDB, downloads all files from GridFS, and re-indexes them). |
| `verify_index` | Runs quick diagnostic checks on MongoDB collections, GridFS files, and ChromaDB vector count. |
| `clear` | Wipes the volatile conversation history and resets user profile records. |
| `exit` / `quit` | Closes the chatbot application. |

---

## 📅 Detailed Phase-by-Phase Implementation

### Phase 1: Environment & Base Configuration
- Setup project framework using `uv` with `pyproject.toml` configuration.
- Created `config.py` environment loading routine to validate `GOOGLE_API_KEY` and construct standardized directories for local data.

### Phase 2: MongoDB & GridFS Document Store
- Implemented `MongoConnection` singleton to manage database sessions securely.
- Built `MetadataManager` to execute CRUD operations on the `documents` collection.
- Created `GridFSManager` to upload and delete binary PDF files.
- Built `DocumentUploader` to coordinate two-phase commits: uploading binary data to GridFS and registering corresponding metadata, rolling back the binary upload if metadata registration fails.
- Wrote `migrate.py` to port legacy local files to MongoDB on setup.

### Phase 3: Conversational Memory & User Profiling
- Created `Session` structure to parse and update user profile metrics (age, conditions, weight, height).
- Formulated `MemoryManager` using LangChain's `ChatMessageHistory` to retain the last 20 conversation turns.
- Injected chat logs and session profiles dynamically into LLM prompt templates.

### Phase 4: Vector Store & RAG Pipeline Setup
- Configured local HuggingFace embeddings (`all-MiniLM-L6-v2`) inside `embeddings.py`.
- Formed the ChromaDB integration in `vectorstore.py` with custom collection namespaces.
- Programmed text segmentation inside `splitter.py` with 1,000-character chunk sizes and 200-character overlaps.
- Setup score-based retrievers to assign human-readable confidence scores (High/Medium/Low) based on vector distance metrics.

### Phase 5: Incremental Indexing & System Orchestration
*   **Phase 5.1 (Incremental Indexer):** Wrote the `Indexer` engine. It scans MongoDB for documents marked `indexed: false`, pulls their bytes from GridFS to a temp file, computes SHA256 hashes for deduplication, splits the content, writes chunks to ChromaDB, and updates the database flag to `indexed: true`.
*   **Phase 5.2 (Diagnostic Tooling):** Built `verify_index` (CLI) and `audit_pipeline.py` (Script) to diagnose the database state, comparing chunk counts and file counts between MongoDB and ChromaDB.
*   **Phase 5.3 (Windows OS Mitigation):** Patched index rebuild failures on Windows systems. Closed and reset the SQLite database connection client, dereferenced collection pointers, and invoked garbage collection (`gc.collect()`) prior to directory deletions.
*   **Phase 5.4 (Custom Source Metadata Filters):** Enabled query parsing to automatically inject source constraints (e.g. `{"source": {"$contains": "WHO"}}`) into vector search queries when users mention specific agencies.

---

## 📈 Current Project Status

The project is currently at **Phase 5.4**. All backend integrations are complete. The CLI application is ready for production tests. Indexing works incrementally, avoiding redundant embedding generation, and operates reliably on Windows systems.

---

## 🗺️ Future Roadmap (Phases 6–10)

*   **Phase 6 (User Authentication & Core Persistence):** Secure multi-user login routing. Store user conversation histories and profile sessions persistently in MongoDB rather than keeping them solely in-memory.
*   **Phase 7 (Hybrid Retrieval Engine):** Blend vector semantic retrieval with BM25 keyword matching (using LangChain's EnsembleRetriever) to improve lookup accuracy for specific terms, abbreviations, and codes.
*   **Phase 8 (Agentic RAG Flow):** Build an active self-corrective routing agent that evaluates retrieval quality, rewrites queries if search results are poor, and falls back to a verified medical web search API.
*   **Phase 9 (Modern GUI & REST API):** Move away from CLI scripts by launching a FastAPI backend and a Next.js/React frontend with streaming chat bubbles, document uploading panels, and visual database indexing monitors.
*   **Phase 10 (HIPAA Compliance & Auditing):** Encrypt personal medical session metadata at rest, run PII scrubbers on user prompts before sending data to external APIs, and generate detailed data audit trails.

---

## 📚 References

- **LangChain Documentation:** [https://python.langchain.com/](https://python.langchain.com/)
- **ChromaDB Vector Database:** [https://www.trychroma.com/](https://www.trychroma.com/)
- **MongoDB GridFS Specs:** [https://www.mongodb.com/docs/manual/core/gridfs/](https://www.mongodb.com/docs/manual/core/gridfs/)
- **Google Gemini API Reference:** [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
- **HuggingFace Sentence Transformers:** [https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
