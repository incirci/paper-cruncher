# Journal Article AI Assistant - Architecture

## Project Structure

```text
cruncher/
├── backend/
│   ├── api/              # FastAPI endpoints
│   │   ├── chat.py       # Chat & session endpoints
│   │   ├── papers.py     # Paper upload & management
│   │   ├── tokens.py     # Token tracking
│   │   ├── agent.py      # Agent config
│   │   ├── mindmap.py    # Mindmap endpoints
│   │   └── config.py     # App config
│   ├── core/             # Configuration
│   │   └── config.py     # Settings loader
│   ├── services/         # Business logic
│   │   ├── ai_agent.py           # Gemini integration
│   │   ├── citation_service.py   # OpenAlex citation graph
│   │   ├── conversation_manager.py  # Session & message management
│   │   ├── mindmap_service.py    # Knowledge graph generation
│   │   ├── openalex_client.py    # OpenAlex API client
│   │   ├── paper_manager.py      # Paper metadata
│   │   ├── pdf_processor.py      # PDF extraction
│   │   ├── progress_manager.py   # SSE progress streaming
│   │   ├── token_tracker.py      # Token monitoring
│   │   └── vector_db.py          # ChromaDB
│   ├── models/           # Data models
│   │   └── schemas.py    # Pydantic schemas
│   └── main.py           # FastAPI app
├── frontend/
│   └── index.html        # Web UI (sessions sidebar, papers sidebar, chat)
├── data/                 # Runtime data (auto-created)
│   ├── uploads/          # Uploaded PDFs
│   ├── vectordb/         # ChromaDB storage
│   └── mindmap/          # Mindmap cache
│       ├── sessions/     # Per-session graphs
│       └── graph.json    # Global graph
├── papers/               # Optional PDF storage (legacy)
├── config.toml           # App configuration
├── .env                  # API keys
└── run.py                # Run script
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
   - Citation service (OpenAlex integration, hierarchical graph building, influence sorting, caching, URL resolution)
   - Conversation management (CRUD, duplication, history)
   - Paper manager (Metadata persistence, pruning logic)
   - Token tracking service
   - Progress manager (SSE-based real-time status updates)

4. **Models Layer** (`backend/models/`)
   - Pydantic schemas and DTOs
   - Paper metadata including a canonical title used consistently across UI, vector DB, and mindmap
   - Message schema now includes `id` for selective deletion
   - Conversation schema includes `notes` for session-specific user remarks

5. **Utils Layer** (`backend/utils/`)
   - Token counters
   - Text chunking helpers
   - Misc helper functions

### Frontend Architecture

The frontend is a single-file application (`index.html`) designed for simplicity and performance, but built with a robust design system:

1. **Unified Design System**:
   - **Color Palette**: "Science Blue" (`#1a73e8`) for primary actions and "Cool Grays" for UI chrome, ensuring high contrast and professional appearance.
   - **Component Library**: Standardized CSS classes (`.btn`, `.btn-primary`, `.btn-secondary`, `.modal-content`) replace ad-hoc styling.
   - **Visual Texture**: A global SVG noise filter (`<feTurbulence>`) is applied via CSS `background-image` to create a subtle, paper-like grain, reducing the "flatness" of standard web UIs.

2. **State Management**:
   - **Server-Driven UI**: The frontend state is a reflection of the backend. Actions (like selecting a paper) trigger API calls, and the UI updates based on the response or SSE events.
   - **Real-time Updates**: `EventSource` connects to `/api/progress` to stream upload status and processing events directly to the UI.

### Data Flow

```text
User Request → FastAPI → Service Layer → AI Agent / Vector DB
                         ↓                  ↓
                  Mindmap Service      ChromaDB (paper_chunks)
                         ↓
                 Gemini (LLM models)
                         ↓
                      Response

User Request (Citation Map) → FastAPI → CitationService → Local Cache / OpenAlex API
                                  ↓
                              Response (Hierarchical JSON with Citation Counts & URLs)
                                  ↓
                              D3.js (Logarithmic Sizing, Influence Sorting, Click Handlers)

User Request (Open PDF) → FastAPI (StaticFiles mount) → data/uploads/{filename} → Browser

Background Tasks (Upload/Reindex) → ProgressManager → SSE Stream (/api/progress) → Frontend UI

Prune Data → PaperManager.prune_papers → Identify unused papers
           → Delete PDFs, Vector Chunks, Metadata
           → OpenAlexClient.clear_cache
           → MindmapService.invalidate_global_cache
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
- **"UI follows backend" consistency**: All state changes sync through the backend; the UI is a reflection of backend state, updated via SSE.
