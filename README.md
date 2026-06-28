# AI-Powered Health Assistant

A production-ready, Supervisor-driven Multi-Agent AI Healthcare Assistant built
with LangGraph, Google Gemini 2.5 Flash, ChromaDB, MongoDB, and Hybrid Search.

---

## Motivation

General-purpose chatbots handle all healthcare questions the same way — through
one generic LLM call. This system separates responsibilities: a Supervisor routes
every query to the right specialist, deterministic tools replace LLM arithmetic,
and Hybrid Search recovers documents that pure semantic search misses (abbreviations,
drug names, organization-specific terminology).

---

## Features

| Feature | Description |
|---|---|
| Multi-Agent Architecture | Supervisor + 5 specialist agents, each with a single responsibility |
| Hybrid Search | ChromaDB vector search + BM25 keyword search merged via Reciprocal Rank Fusion |
| Knowledge Retrieval Tool | Reusable, cacheable semantic + keyword document retrieval |
| Medical Calculator Tool | Deterministic BMI, BMR, water intake, IBW, BSA — no LLM arithmetic |
| Document Analysis Tool | PDF structural analysis, doc-type classification, clinical finding detection |
| System Status Tool | Real-time health checks for all backend components |
| Conversation Memory | Per-session chat history injected into every prompt |
| Structured Logging | Per-node execution times, tool logs, full request lifecycle |
| Evaluation Metrics | Aggregate retrieval/LLM/workflow timing from `metrics.json` |
| Hybrid Search Benchmark | 40-query benchmark comparing Vector-only vs Hybrid accuracy |
| 220+ Automated Tests | Covering routing, tools, hybrid search, error handling, performance |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| Workflow Orchestration | LangGraph |
| Vector Store | ChromaDB (persistent, local) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace / sentence-transformers) |
| Keyword Search | rank-bm25 (BM25Okapi) |
| Document Storage | MongoDB + GridFS |
| PDF Processing | pypdf |
| Language | Python 3.11+ |

---

## Architecture

### Overall Workflow

```
User Input
    │
    ▼
LangGraph Workflow
    │
    ▼
Input Node  ──►  Intent Detection Node
                       │
                       ▼
               Supervisor Agent
               (selects specialist)
                       │
          ┌────────────┴────────────┐
          ▼ (parallel)              ▼
    Memory Node              Retriever Node
    (chat history)           (Hybrid Search)
          │                        │
          └────────────┬───────────┘
                       ▼
                  Tool Node
          (Calculator / DocAnalysis / skip)
                       │
                       ▼
             Agent Execution Node
          (Document QA / Drug / Nutrition /
           Mental Health / General Health)
                       │
                       ▼
               Formatter Node
          (citations + hybrid metrics)
                       │
                       ▼
                  Response → User
```

### Multi-Agent Architecture

```
Supervisor Agent
    │
    ├── Document QA Agent      ← medical questions, grounded in documents
    ├── Drug Information Agent ← medicines, dosage, interactions
    ├── Nutrition Agent        ← diet, meal plans, vitamins, BMI/BMR
    ├── Mental Health Agent    ← stress, anxiety, sleep (educational only)
    └── General Health Agent   ← greetings, wellness, small-talk
```

### Tool Architecture

```
Tool Layer
    │
    ├── Knowledge Retrieval Tool
    │       ├── Vector Search (ChromaDB)
    │       ├── BM25 Search   (rank-bm25)
    │       ├── RRF Merge
    │       └── Content Deduplication
    │
    ├── Medical Calculator Tool
    │       ├── BMI  (WHO formula)
    │       ├── BMR  (Mifflin–St Jeor)
    │       ├── Water Intake (35 mL/kg)
    │       ├── Ideal Body Weight (Devine)
    │       └── Body Surface Area (Mosteller)
    │
    ├── Document Analysis Tool
    │       ├── PDF text extraction (pypdf)
    │       ├── Document type classification (regex)
    │       └── Clinical finding detection (regex patterns)
    │
    └── System Status Tool
            ├── MongoDB ping
            ├── GridFS accessibility
            ├── ChromaDB chunk count
            ├── LangGraph compilation check
            ├── Gemini API key validation
            └── BM25 availability
```

### Hybrid Search

```
User Query
    │
    ├──► ChromaDB Vector Search  ──► top-K semantic chunks
    └──► BM25 Keyword Re-rank    ──► top-K keyword-ranked chunks
                │
                ▼
     Reciprocal Rank Fusion (RRF)
                │
                ▼
     Content Deduplication (first-200-char fingerprint)
                │
                ▼
     Final Ranked Results  →  LLM Prompt
```

**Why Hybrid Search?**

| Query Type | Vector-only | Hybrid |
|---|---|---|
| `What causes hypertension?` | ✅ | ✅ |
| `HbA1c level in diabetes` | ❌ | ✅ |
| `WHO hypertension guideline` | ❌ | ✅ |
| `LDL HDL cholesterol` | ❌ | ✅ |
| `CDC cardiovascular statistics` | ❌ | ✅ |

Hybrid Search improves recall for abbreviations, drug names, and
organization-specific queries where semantic similarity alone is insufficient.

### Database Design

```
MongoDB
  └── health_assistant (database)
        └── documents (collection)
              ├── _id           ObjectId
              ├── title         str
              ├── filename      str
              ├── category      str
              ├── organization  str
              ├── indexed       bool
              ├── chunk_count   int
              ├── file_hash     str   (SHA-256, deduplication)
              └── gridfs_id     ObjectId → GridFS binary

GridFS
  └── Stores raw PDF binary data

ChromaDB  (local persistent)
  └── medical_documents (collection)
        └── Embeddings + metadata per chunk
              Chunk ID: "{doc_id}_page{page}_chunk{n}"
```

---

## Folder Structure

```
HealthAssistant/
├── agents/
│   ├── supervisor_agent.py        # Central router
│   ├── document_agent.py          # Medical Q&A
│   ├── drug_agent.py              # Pharmacology
│   ├── nutrition_agent.py         # Diet & nutrition
│   ├── mental_health_agent.py     # Wellness guidance
│   ├── general_health_agent.py    # Greetings & wellness
│   └── formatter_agent.py         # Response formatting
├── graph/
│   ├── builder.py                 # LangGraph compilation
│   ├── nodes.py                   # All 8 workflow nodes
│   ├── edges.py                   # Edge definitions
│   ├── router.py                  # Conditional routing
│   ├── state.py                   # Shared state TypedDict
│   ├── workflow.py                # Workflow wrapper
│   ├── cache.py                   # In-memory retrieval cache
│   ├── logger.py                  # Structured logging
│   ├── metrics.py                 # metrics.json export
│   ├── monitor.py                 # Startup diagnostics
│   └── visualizer.py              # Mermaid export
├── tools/
│   ├── knowledge_retrieval_tool.py  # Hybrid search tool
│   ├── calculator_tool.py           # BMI/BMR/water/IBW/BSA
│   ├── document_analysis_tool.py    # PDF analysis
│   ├── system_status_tool.py        # Health checks
│   ├── tool_result.py               # Standardised ToolResult
│   └── retrieval_tool.py            # Backward-compat shim
├── rag/
│   ├── pipeline.py                # RAGPipeline orchestrator
│   ├── vectorstore.py             # ChromaDB interface
│   ├── retriever.py               # Retriever + formatter
│   ├── indexer.py                 # Incremental indexing
│   ├── loader.py                  # PDF loader
│   ├── splitter.py                # Text chunking
│   └── embeddings.py              # HuggingFace embeddings
├── database/
│   ├── connection.py              # MongoDB singleton
│   ├── document_service.py        # High-level document API
│   ├── gridfs_manager.py          # GridFS operations
│   ├── metadata.py                # Document metadata CRUD
│   └── uploader.py                # Upload orchestration
├── memory/
│   ├── memory_manager.py          # Chat history manager
│   └── session.py                 # User session profile
├── prompts/
│   ├── system_prompt.py           # Base system prompt
│   └── templates.py               # LangChain prompt templates
├── utils/
│   ├── evaluation_metrics.py      # Aggregate metrics from JSON
│   └── benchmark.py               # 40-query hybrid benchmark
├── tests/
│   └── test_final_validation.py   # 100-check final test suite
├── app.py                         # CLI entry point
├── config.py                      # Configuration & env vars
├── metrics.json                   # Auto-generated per-request metrics
├── benchmark_results.json         # Benchmark output (after running)
├── workflow.mmd                   # Mermaid workflow diagram
├── test_phase7.py                 # Phase 7 tests  (120 checks)
├── test_phase71.py                # Phase 7.1 tests (45 checks)
├── test_phase72.py                # Phase 7.2 tests (120 checks)
├── .env.example                   # Environment variable template
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- MongoDB running locally (`mongodb://localhost:27017`) or MongoDB Atlas
- Google Gemini API key

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd HealthAssistant

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY and MONGODB_URI
```

---

## Configuration

Edit `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=health_assistant
APP_ENV=development
```

All other settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model |
| `LLM_TEMPERATURE` | `0.3` | Response determinism |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding |
| `RETRIEVER_TOP_K` | `6` | Chunks retrieved per query |
| `RETRIEVER_DISTANCE_THRESHOLD` | `1.25` | Max ChromaDB distance |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

---

## Usage

### Start the assistant

```bash
python app.py
```

### Available commands

| Command | Description |
|---|---|
| `status` | Display full system diagnostics (MongoDB, GridFS, ChromaDB, Gemini, BM25) |
| `docs` | List all uploaded medical documents |
| `upload` | Upload a new medical PDF |
| `verify_index` | Verify index health (duplicates, ghost docs, chunk counts) |
| `rebuild` | Force full index rebuild |
| `clear` | Reset conversation memory |
| `exit` | Quit |

### Example session

```
User: Hello
Assistant: (💬 General Health Agent) Hello! I'm your AI Health Assistant...

User: What causes hypertension?
Assistant: (📄 Document QA Agent, 🔍 Knowledge Retrieval Tool) Hypertension...
  Sources: CDC — About High Blood Pressure (Page 2) · Confidence: High

User: My height is 181 cm and weight is 52 kg. Calculate my BMI.
Assistant: (🥗 Nutrition Agent, 🧮 Medical Calculator BMI)
  BMI: 15.9  |  Category: Mild Thinness / Underweight

User: Can I take Aspirin with Ibuprofen?
Assistant: (💊 Drug Information Agent) These two NSAIDs compete...

User: status
  ============================================
            System Diagnostics
  ============================================
  MongoDB              : Healthy  (3ms)  — 51 documents
  GridFS               : Healthy  (1ms)  — 51 files
  ChromaDB             : Healthy  (2ms)  — 6061 chunks
  LangGraph            : Compiled (0ms)  — 10 nodes
  Gemini               : Connected (0ms) — gemini-2.5-flash
  Hybrid Search        : Available       — BM25 enabled
  ============================================
  Documents            : 51
  Vector Chunks        : 6061
  Agents               : 6
  Tools                : 4
  ============================================
  Overall              : Healthy ✓
  Execution Time       : 14ms
  ============================================
```

---

## Testing

### Run all test suites

```bash
# Phase 7 — Multi-agent routing (120 checks)
.venv\Scripts\python.exe test_phase7.py

# Phase 7.1 — Tool layer (45 checks)
.venv\Scripts\python.exe test_phase71.py

# Phase 7.2 — Hybrid search (120 checks)
.venv\Scripts\python.exe test_phase72.py

| Phase 7.3 — Final validation (100 checks) | ✅ 100/100 |
| **Total** | **283/283** |

The benchmark suite (`utils/benchmark.py`) runs 40 medical queries and produces a
comparison table showing Hybrid Search vs Vector-only accuracy.

### Run the hybrid search benchmark (requires live database)

```bash
.venv\Scripts\python.exe utils/benchmark.py
```

This runs 40 medical queries against the live ChromaDB, compares Vector-only vs
Hybrid retrieval, and saves results to `benchmark_results.json`.

### View evaluation metrics

```python
from utils.evaluation_metrics import EvaluationMetrics
em = EvaluationMetrics()
summary = em.compute(doc_count=51, chunk_count=6061)
print(summary)
```

---

## Performance

Measured on a local development machine with 51 documents / 6061 chunks:

| Operation | Average Time |
|---|---|
| Intent Detection | ~0.8 s |
| Hybrid Search (Vector + BM25) | ~150 ms |
| Calculator Tool | < 1 ms |
| Document Analysis Tool | ~400 ms |
| Gemini LLM Response | ~2.4 s |
| Total Workflow | ~3.4 s |
| Cache Hit (retrieval) | ~5 ms |

---

## Hybrid Search Benchmark Results

Run `utils/benchmark.py` to generate live numbers. Expected results with 51 indexed documents:

| Metric | Value |
|---|---|
| Total Queries | 40 |
| Vector-only Accuracy | ~80% |
| Hybrid Accuracy | ~93% |
| Improvement | +~13% |
| Average Vector Search Time | ~120 ms |
| Average Hybrid Search Time | ~145 ms |
| Cache Hit Rate | ~30% |

**Category breakdown:**

| Category | Queries | Vector | Hybrid |
|---|---|---|---|
| General Medical | 10 | High | High |
| Abbreviations (HbA1c, LDL…) | 10 | Low | High |
| Nutrition | 10 | High | High |
| Mental Health | 5 | High | High |
| Org-Specific (WHO/CDC) | 5 | Low | High |

Hybrid search has the largest advantage for **abbreviation** and
**organization-specific** queries where BM25 keyword matching outperforms
pure semantic similarity.

**Recent benchmark output:**

```
======================================================================
  Hybrid Search Benchmark Results  –  Phase 7.3
======================================================================
  Total Queries   : 40
  Vector Accuracy : 80.0%
  Hybrid Accuracy : 92.5%
  Improvement     : +12.5%
  Vector Avg Time : 122 ms
  Hybrid Avg Time : 141 ms
======================================================================

  #   Category       Query                                    Vector   Hybrid
  ----------------------------------------------------------------------

  1   Medical        What causes hypertension?                ✅       ✅
  2   Abbreviation   HbA1c level in diabetes                  ❌       ✅
  3   Abbreviation   WHO hypertension guideline               ❌       ✅
  4   Nutrition      Foods rich in iron                       ✅       ✅
  5   Medical        What is coronary artery disease?         ✅       ✅
  ...
  40  Org-Specific   CDC statistics on heart disease          ❌       ✅
  ----------------------------------------------------------------------
  TOTAL            32/40 (80%)                            37/40 (92.5%)
======================================================================
```

---

## Limitations

- **No real-time data** — answers are grounded in uploaded documents; the
  system does not search the internet.
- **Educational only** — all agents explicitly state they do not provide
  diagnoses, prescriptions, or professional medical advice.
- **Single-user CLI** — designed as a command-line chatbot; no web frontend.
- **Local BM25** — BM25 re-ranks the vector top-K only; it does not search
  the full corpus independently. A future version could use a full-text index
  (Elasticsearch / MongoDB Atlas Search) for broader keyword recall.
- **Memory is in-session** — conversation history is not persisted across
  application restarts (uses LangChain MemorySaver in-process checkpointer).

---

## Future Scope

The architecture is designed to accept new agents and tools without redesigning
the core workflow. Planned extensions (not in scope for this release):

- **Medical Report Analysis Agent** — CBC, liver function, kidney function
- **OCR Agent** — prescription and lab report text extraction
- **Health Risk Prediction Agent** — diabetes, hypertension, heart disease risk
- **Full-corpus BM25** — Elasticsearch or Atlas Search backend behind the
  Knowledge Retrieval Tool interface
- **Web UI** — Streamlit or FastAPI frontend
- **Persistent memory** — MongoDB-backed conversation history

---

## Demo Sequence

The following 8-step sequence demonstrates every major component:

```
1.  status                                    → System Status Tool
2.  Hello                                     → General Health Agent
3.  What causes hypertension?                 → Document QA + Hybrid Search
4.  Calculate BMI Height 181 cm Weight 52 kg  → Nutrition Agent + Calculator Tool
5.  Foods rich in iron                        → Nutrition Agent + Hybrid Search
6.  Can I take Aspirin with Ibuprofen?        → Drug Information Agent
7.  Analyze my uploaded report                → Document QA + Document Analysis Tool
8.  verify_index                              → Index verification
```
