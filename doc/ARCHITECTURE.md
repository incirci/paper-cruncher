# Journal Article AI Assistant - Architecture

## Project Structure

```
cruncher/
├── backend/
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Configuration & settings
│   ├── services/         # Business logic
│   ├── models/           # Data models & schemas
│   └── utils/            # Helper functions
├── frontend/             # Web UI
├── data/                 # Vector DB & conversations (gitignored)
├── papers/               # PDF storage (gitignored)
├── tests/                # Test suite
├── config.toml           # Application config
├── .env                  # Environment variables (gitignored)
├── .env_example          # Environment template
└── requirements.txt      # Python dependencies
```

## Architecture Overview

### Backend Components

1. **API Layer** (`backend/api/`)
   - FastAPI endpoints for REST API
   - Request/response models
   - Route handlers
   - Mindmap API (`/api/mindmap`) and papers API (`/api/papers`) for knowledge graph rebuild and retrieval

2. **Core Layer** (`backend/core/`)
   - Configuration management
   - Settings loaders
   - Application initialization

3. **Services Layer** (`backend/services/`)
   - PDF processing service (extracts text, metadata, and chunks)
   - Vector database service (ChromaDB wrapper, semantic search, paper micro-summaries)
   - AI agent service (orchestrator-worker RAG)
   - Mindmap service (LLM-based graph generation + post-processing)
   - Citation service (OpenAlex integration, hierarchical graph building)
   - Conversation management
   - Token tracking service

4. **Models Layer** (`backend/models/`)
   - Pydantic schemas and DTOs
   - Paper metadata including a canonical title used consistently across UI, vector DB, and mindmap

5. **Utils Layer** (`backend/utils/`)
   - Token counters
   - Text chunking helpers
   - Misc helper functions

### Data Flow

```text
User Request → FastAPI → Service Layer → AI Agent / Vector DB
                         ↓                  ↓
                  Mindmap Service      ChromaDB (paper_chunks)
                         ↓
                 Gemini (LLM models)
                         ↓
                      Response

User Request (Citation Map) → FastAPI → CitationService → OpenAlex API
                                  ↓
                              Response (Hierarchical JSON)
```

### Orchestrator-Worker RAG & Mindmap Pipeline

```text
User Query → Orchestrator (plans commands) → Worker (executes retrieval) →
Context Building → Prompt → Gemini API (streaming) → Response

Index Papers → PDF Processor → VectorDBService.add_paper_chunks
             → VectorDBService.get_paper_summaries (micro-summaries)
             → [Optional: Vector Search for Custom Query]
             → MindmapService.build_prompt (concept-focused + optional RAG context)
             → Gemini (graph JSON) → MindmapService._normalize_and_deduplicate
             → data/mindmap/graph.json → D3.js viewer (/mindmap)
```

### Deep Dive Retrieval

The Orchestrator can now dynamically adjust retrieval density based on query complexity:

- **Normal Density**: Fetches standard number of chunks (default: 10).
- **High Density ("Deep Dive")**: Fetches 4x chunks (default: 40) for complex queries requiring comprehensive detail.
- Controlled via `DENSITY` parameter in the Orchestrator's system prompt.

### RAG-Powered Mindmaps

The Mindmap generation pipeline now supports Retrieval Augmented Generation (RAG) for custom queries:

1. **User Query**: User provides specific instructions (e.g., "Map out all sensors used").
2. **Vector Search**: System searches the vector DB for chunks relevant to the query.
3. **Context Injection**: Retrieved chunks are injected into the LLM prompt alongside paper summaries.
4. **Generation**: The LLM uses both high-level summaries and specific details to build the tree.

## Technology Stack

- **Backend**: FastAPI + Python 3.12
- **AI Model**: Google Gemini 2.5 Flash
- **External APIs**: OpenAlex (Citation Data)
- **Vector DB**: ChromaDB
- **Database**: SQLite (conversations)
- **PDF Processing**: PyMuPDF
- **Configuration**: `tomllib` (Python 3.11+ standard library)
- **Visualization**: D3.js (CDN) for the mindmap viewer

## Key Design Principles

- Simple, clean code
- Modern Python features (Python 3.12+, `tomllib`)
- No legacy support
- Token optimization first
- Testable architecture
- Prompt engineering for better AI responses
- Orchestrator-Worker is the single retrieval mode (no toggles)
- Canonical paper titles used consistently across backend and frontend
- Mindmap generation is deterministic, concept-focused, and post-processed for normalization
