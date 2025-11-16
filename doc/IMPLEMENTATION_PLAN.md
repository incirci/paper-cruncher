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
- [x] Backend: Introduce canonical paper titles in `PaperMetadata` and propagate to vector DB, sidebar, and mindmap leaves
- [x] Backend: Implement keyword-biased micro-summaries in `VectorDBService.get_paper_summaries` for concept-rich paper descriptions
- [x] Backend: Tighten `MindmapService.build_prompt` to focus internal node names on conceptual content and avoid generic section/review labels
- [x] Backend: Add post-processing (`_normalize_and_deduplicate`) to merge duplicate internal concept nodes while preserving paper leaves

### Custom Mindmaps (User-Defined Instructions)

Add support for user-editable mindmap instructions ("mindmap query") to generate alternative tree structures without changing the core architecture.

Completed tasks:

- [x] API: Extend `GET /api/mindmap` to accept an optional `query` parameter that carries user-provided mindmap instructions.
- [x] Service: Update `MindmapService.generate_graph` / `build_prompt` to accept an optional custom query and include it in the LLM prompt while preserving all structural constraints.
- [x] Persistence rules: Only persist the default (no custom query) global mindmap to `data/mindmap/graph.json`; treat custom mindmaps as ephemeral responses.
- [x] Frontend (sidebar): Add an edit icon near the "View Mindmap" button that opens a prompt with the default mindmap instructions, scoped to either all papers or the selected paper.
- [x] Frontend (mindmap page): Read both `paper_id` and `query` from URL parameters and forward them to `/api/mindmap` when loading the tree.

### ChatGPT-Style UX Enhancements

Add a small set of high-impact UI improvements inspired by the ChatGPT interface.

Planned tasks:

- [ ] File/image upload: Add an upload control to the chat input area to attach files/images to the current session and expose them to the backend (initially as metadata and temporary storage, without automatic indexing).
- [ ] Scroll-to-bottom button: Show a floating "scroll to latest" button when the user scrolls up in a long conversation; clicking jumps to the newest message and re-enables auto-scroll.
- [ ] New Chat: Add a "New Chat" button that clears the visible chat history and requests a new backend `session_id`, preserving the currently selected paper unless changed by the user.
- [ ] Per-code-block copy: Enhance markdown rendering so each code block includes its own copy button that copies just that block's content.

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
