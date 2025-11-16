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

2. **Core Layer** (`backend/core/`)
   - Configuration management
   - Settings loaders
   - Application initialization

3. **Services Layer** (`backend/services/`)
   - PDF processing service
   - Vector database service
   - AI agent service (Orchestrator-Worker)
   - Conversation management
   - Token tracking service

4. **Models Layer** (`backend/models/`)
   - Pydantic schemas
   - Database models
   - Data transfer objects

5. **Utils Layer** (`backend/utils/`)
   - Token counters
   - Text chunking
   - Helper functions

### Data Flow

```
User Request → FastAPI → Service Layer → AI Agent/Vector DB
                    ↓
              Response ← Gemini API ← Context Builder
```

### Orchestrator-Worker RAG Pipeline

```
User Query → Orchestrator (plans commands) → Worker (executes retrieval) →
Context Building → Prompt → Gemini API (streaming) → Response
```

## Technology Stack

- **Backend**: FastAPI + Python 3.11+
- **AI Model**: Google Gemini 2.5 Pro
- **Vector DB**: ChromaDB
- **Database**: SQLite (conversations)
- **PDF Processing**: PyMuPDF

## Key Design Principles

- Simple, clean code
- Modern Python features
- No legacy support
- Token optimization first
- Testable architecture
- Prompt engineering for better AI responses
- Orchestrator-Worker is the single retrieval mode (no toggles)
- Structured output formatting via system instructions
