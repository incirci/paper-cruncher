# Implementation Plan - Concise Version

## Project Status

**Core Application: ✅ COMPLETE**

All essential features are implemented and working:

- ✅ Web UI with collapsible sidebar
- ✅ PDF processing and manual indexing
- ✅ Vector database with semantic search (ChromaDB)
- ✅ Google Gemini 2.5 Pro AI agent
- ✅ RAG pipeline with orchestrator-worker architecture
- ✅ Conversation history with persistence
- ✅ Token tracking and monitoring
- ✅ Markdown formatting in responses
- ✅ Persistent paper metadata across restarts

## Quick Start

```bash
# Windows
.venv\Scripts\python.exe run.py

# Open browser
http://localhost:8000
```

## Usage Flow

1. Add PDFs to `papers/` folder
2. Click "📚 Index Papers" button (one-time)
3. Chat with AI about your papers
4. Papers persist - no re-indexing needed unless adding new papers

## Architecture Decisions

### Current Implementation

- **Retrieval**: Orchestrator-worker pattern (single path)
- **Storage**: ChromaDB for vectors, SQLite for conversations
- **Frontend**: Vanilla JS served by FastAPI
- **Markdown**: Marked.js for formatting
- **Memory**: Configurable session persistence

### Key Features

- Manual indexing control (user-triggered)
- Streaming responses for better UX
- Token optimization through smart chunking
- Context-aware conversation history (last 10 messages)

## Knowledge Graph & Mindmap (New)

Add an AI-generated hierarchical knowledge structure to organize topics and papers, visualized as a collapsible tree (NotebookLM-style). Support paper-specific context selection for scoped chat and mindmap.

### Scope

- Build hierarchical topic structure with papers as leaf nodes (global view)
- Build paper-specific hierarchical structure (paper as root, topics as children)
- Save global JSON in tree format to `data/mindmap/graph.json`
- Generate paper-scoped mindmaps on-demand (not persisted)
- Visualize at `GET /mindmap` using D3.js collapsible tree (CDN, no build tools)
- Serve data via `GET /api/mindmap?paper_id=<id>` (optional param for paper-scoped view)
- Add paper selection UI in sidebar (click to select/unselect)
- Context-aware chat: when paper selected, scope all questions to that paper only

### Implementation Tasks

- [x] UI: Add paper selection/deselection in sidebar with visual indication
- [x] UI: Pass selected paper_id to chat API and mindmap API
- [x] Service: `MindmapService` - add method to generate paper-scoped tree from single paper
- [x] API: `GET /api/mindmap?paper_id=<id>` - return global or paper-scoped tree
- [x] API: `POST /api/chat` - accept optional paper_id, scope retrieval to selected paper
- [x] UI: `/mindmap` page - fetch with paper_id if selected, render accordingly
- [x] Service: Update AI agent to handle paper_id filter for focused retrieval

## Remaining Optional Enhancements

### Testing (Phase 5)

- [ ] Unit tests for services
- [ ] Integration tests
- [ ] API documentation (Swagger)

### Features (Nice-to-have)

- [ ] Conversation summarization for long chats
- [ ] Response caching for identical queries
- [ ] WebSocket support
- [ ] Agent profile editing UI
- [ ] Mobile responsive design
- [ ] Clear history button in UI
- [ ] Settings panel
- [ ] Message timestamps

### Performance (Phase 6)

- [ ] Application profiling
- [ ] Database query optimization
- [ ] Request caching
- [ ] Rate limiting

## Configuration

### config.toml

Key settings:

- `papers_folder`: Where PDFs are stored
- `agent.model`: AI model (gemini-2.5-flash-preview-09-2025)
- `agent.max_context_tokens`: Context window (1M)
- `chunking.max_chunks_per_query`: Retrieval limit (10)
- `memory.persist_across_sessions`: Session memory (true)

### .env

Required:

- `GOOGLE_API_KEY`: Your Gemini API key

## Success Criteria

All core requirements met:

- ✅ AI answers questions about individual papers
- ✅ AI synthesizes information across multiple papers
- ✅ Conversation history persists (configurable)
- ✅ Token usage tracked accurately
- ✅ Responsive and intuitive UI
- ✅ Production-ready error handling and logging

## Code Quality

Follows instruction requirements:

- ✅ Simple, concise code
- ✅ Well-structured architecture
- ✅ No redundancy
- ✅ Modern Python features
- ✅ No legacy fallbacks
- ✅ Clean, forward-focused code

---

**Note**: Application is fully functional for research paper analysis. Optional enhancements listed above are for future improvements but not required for core functionality.
