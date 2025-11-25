# Journal Article AI Assistant

An AI-powered web application for analyzing and querying journal articles using Google Gemini 2.5 Flash with RAG (Retrieval Augmented Generation).

## Quick Start

### 1. Prerequisites

- Python 3.12+ (required for `tomllib`)
- Google Gemini API key

### 2. Setup (Local)

```bash
# Install dependencies (already done if you followed initial setup)
pip install -r requirements.txt

# Create .env file with your API key
# (Copy from .env_example and add your key)
```

### 3. Add Papers

**Recommended:** Use the web interface to upload papers.

1. Start the application.
2. Click "📎 Upload Papers" in the sidebar.
3. Select PDF files. They will be saved to `data/uploads/` and indexed automatically.

**Legacy:**
Placing papers in the `papers/` folder is supported for backward compatibility but requires manual re-indexing.

### 3. Run the Application

**Local:**

```bash
python run.py
```

**Docker:**

See the [Docker](#docker) section below for containerized deployment instructions.

The application will start at `http://localhost:8000`

## Docker

You can run the app in Docker without installing Python locally.

### Build Image

```bash
docker build -t cruncher:latest .
```

### Run Container

```bash
# Replace YOUR_API_KEY with your Google API key
docker run --rm -p 8000:8000 \
   -e GOOGLE_API_KEY=YOUR_API_KEY \
   -v %CD%/papers:/app/papers \
   -v %CD%/data:/app/data \
   -v %CD%/config.toml:/app/config.toml:ro \
   --name cruncher \
   cruncher:latest
```

On PowerShell, ensure the `%CD%` paths are correct; on macOS/Linux use `$PWD`.

### Docker Compose

Create a `.env` file with your API key:

```bash
echo GOOGLE_API_KEY=YOUR_API_KEY > .env
```

Then run:

```bash
docker compose up --build
```

The app will be available at `http://localhost:8000`.

## How to Use

### Adding Papers

**Upload via Sidebar:**

1. Click the "📎 Upload Papers" button in the sidebar
2. Select one or more PDF files from your computer
3. Papers are automatically:
   - Saved to `data/uploads/` directory
   - Indexed into the vector database
   - Available immediately for chat and mindmap
4. Uploaded papers persist across restarts
5. **Open PDF**: Click the file icon (📄) next to any paper in the list to open the PDF in a new browser tab.

### Managing Sessions

**Sessions Sidebar:**

1. Click the "💬 Sessions" button to open the sessions sidebar (opens on the right)
2. View all your saved sessions with:
   - Session ID (truncated)
   - Message count
   - Token usage
   - Last updated timestamp
3. Click any session to switch to it (loads full chat history and papers)
4. Delete sessions using the ✕ button next to each session
5. Sessions are sorted by most recent activity

**Selective Message Deletion:**

- Hover over any message (user or assistant) to reveal the trash icon (🗑).
- Click the icon to delete that specific message from the conversation history.
- This is useful for removing erroneous queries or cleaning up the context window.

**Duplicate Session:**

1. Click the "Duplicate Session" button (copy icon) on any session in the sidebar
2. Creates a new session with the same selected papers
3. Does NOT copy the chat history (starts fresh with the same context)
4. Useful for branching investigations from the same set of papers

**New Session:**

1. Click "New Session" button in the header
2. Creates a fresh conversation with new session ID
3. Preserves currently selected papers unless you change them
4. Previous session is automatically saved to history

**Session Notes:**

- Click the "📝 Notes" button in the header to open a scratchpad for the current session.
- Notes are saved automatically per session.
- Useful for jotting down key insights, to-do lists, or summaries while chatting.

**Session Behavior:**

- All sessions are visible in the sidebar
- Adding/removing papers does NOT clear conversation history
- Each session maintains independent:
  - Chat messages
  - Paper context (selected papers)
  - Token usage tracking

### Asking Questions

1. **Session-Scoped**: Questions are always within current session context
   - AI searches only papers in the current session
   - Each session has its own paper context and conversation history

2. **Global Mode** (no paper selected): Ask questions about all papers in session
   - Type your question in the input box at the bottom
   - AI will search across all papers in the current session
   - Example: "What are the common themes across these papers?"

3. **Paper-Specific Mode**: Focus on a single paper
   - Click any paper in the sidebar to select it (blue highlight)
   - Ask questions scoped to that paper only
   - Example: "What methodology was used in this study?"
   - Click the paper again to deselect and return to global mode

### Viewing the Knowledge Graph

1. Click the "🧠 View Mindmap" button in the sidebar
2. **Session-Scoped Mindmap**: Shows only papers from current session
   - Papers organized by themes and subtopics
   - Click nodes to expand/collapse
   - Validates that selected paper (if any) belongs to session
3. **Paper-Specific Mindmap**: When a paper is selected
   - Shows the selected paper as root
   - Displays main topics and subtopics from that paper
   - Use "Expand All" and "Collapse All" buttons to control view

4. **Custom Mindmaps (User Instructions)**
   - Click the ✏️ icon next to "🧠 View Mindmap"
   - Enter **mindmap instructions** (free-text query)
   - Example: "Create only two top-level themes, one for physical fatigue and one for stress."
   - **Deep Search (RAG-Powered)**: When custom instructions are provided, the system performs a semantic search across all papers to find specific details, equipment, or methods mentioned in your query, ensuring the mindmap is as detailed as a chat response.
   - Instructions influence hierarchy organization while maintaining structure
   - Structural guarantees: valid JSON, canonical paper titles as leaves, depth limits

5. **Mindmap Caching**
   - Session mindmaps are cached on disk per session
   - Reusing same session + query = instant load (no regeneration)
   - Cache survives app restarts
   - Cache location: `data/mindmap/sessions/<session_id>/`

   **Example instructions to try:**

   - "Create only two top-level themes: one for physical fatigue and one for stress, and group all papers accordingly."
   - "Build a shallow two-level map that focuses only on application contexts, with papers grouped under their main real-world use cases."
   - "Emphasize methodological differences: group papers first by type of modeling or analysis approach, then list the papers under each approach."

### Viewing the Citation Map

1. **Access**: Click the "🔗 Citations" button (or similar icon) on a specific paper in the sidebar.
2. **Data Source**: Powered by the **OpenAlex API** to fetch real-world citation data.
3. **Structure**:
   - **Root**: The selected paper.
   - **Branches**: "References" (backward) and "Cited By" (forward).
   - **Grouping**: Papers are automatically grouped by **Research Topic** (e.g., "Artificial Intelligence", "Public Health").
4. **Influence Highlighting**:
   - **Node Sizing**: Papers and Topics are sized logarithmically based on their citation count. Larger nodes = more influential papers.
   - **Configurable**: Min/max node sizes can be adjusted in `config.toml`.
   - **Sorting**: Topics are sorted by total citation impact, and papers within topics are sorted by their individual citation count.
5. **Interactivity**:
   - **Clickable Nodes**: Clicking a paper node opens its source URL (DOI or OpenAlex page) in a new tab.
   - **Tooltips**: Hover over nodes to see full titles and exact citation counts.
6. **Filtering**:
   - **Year Filter**: Filter references and citations by publication year range.
   - **Auto-Detection**: The filter inputs automatically default to the min/max years found in the data.
7. **Integration**:
   - **Local Resolution**: If a cited paper exists in your local library, it is highlighted in green.
   - **Caching**: Citation data is cached locally to ensure fast loading on subsequent visits.

### Pruning Data

1. Click the "✂ Prune Data" button in the sidebar.
2. This action removes all papers that are **not** currently part of any active session.
3. It performs a deep clean, removing:
   - PDF files from disk
   - Vector embeddings from the database
   - Metadata entries
   - Cached citation data (OpenAlex)
   - Global mindmap caches
4. Use this to free up space and keep your knowledge base focused on active projects.

### Features

- **Modern Chat Interface**: ChatGPT-style UI with floating input, auto-expanding text area, and state-aware blocking.
- **Professional UI/UX**:
  - **Science Blue Theme**: Unified color palette (#1a73e8) for a clean, academic aesthetic.
  - **Paper-like Texture**: Subtle SVG noise background for reduced eye strain and organic feel.
  - **Unified Design System**: Consistent button styles and visual hierarchy across all views.
- **Session Management**: Independent chat sessions with dedicated sidebar
  - View all sessions with metadata (messages, tokens, timestamps)
  - Switch between sessions seamlessly
  - Session-scoped paper context
  - Messages persist when papers are added/removed
- **Paper Upload**: Drag-and-drop or select PDFs directly in sidebar
  - Auto-indexing on upload
  - Papers saved to persistent `data/uploads/` directory
  - Session-aware uploads (merge into current session)
- **Conversation History**: Chat maintains context across messages
- **Selective Deletion**: Delete individual messages to refine context
- **Source Citations**: AI references which papers it's using
- **Streaming Responses**: See answers as they're generated
- **Markdown Formatting**: Responses include tables, lists, and code blocks
- **Mermaid Diagrams**: Auto-renders flowcharts and graphs in responses
- **Token Tracking**: Monitor API usage in real-time per session (includes hidden orchestration costs)
- **Canonical Paper Titles**: Consistent titles across UI, DB, and mindmaps
- **Smart Mindmaps**:
  - Session-scoped knowledge graphs
  - **RAG-enhanced custom generation** (searches paper content for specific queries)
  - Keyword-biased micro-summaries
  - Normalized and deduplicated concept nodes
  - Per-session disk caching for instant reuse
  - Custom instructions for alternative views
- **UI Consistency**: "Backend first, UI follows" architecture
  - All state changes sync through backend
  - Real-time session updates via SSE events
  - No UI drift from backend state
- **Data Pruning**: Keep your workspace clean
  - "Prune Data" button removes papers not used in any active session
  - Deep cleans vector DB, metadata, and caches

## Project Structure

For a detailed breakdown of the architecture, services, and data flow, please refer to [ARCHITECTURE.md](doc/ARCHITECTURE.md).

```text
cruncher/
├── backend/              # FastAPI application
├── frontend/             # Single-file HTML/JS UI
├── data/                 # Runtime storage (VectorDB, uploads)
├── config.toml           # Configuration
└── run.py                # Entry point
```

## API Endpoints

### Chat & Sessions

- `POST /api/chat` - Send message to AI agent (returns JSON)
- `POST /api/chat/stream` - Send message to AI agent (returns SSE stream)
- `GET /api/chat/sessions` - List all sessions with metadata
- `GET /api/chat/session/{session_id}` - Get session details
- `POST /api/chat/session` - Create new session
- `POST /api/chat/session/{session_id}/duplicate` - Duplicate session (papers only)
- `DELETE /api/chat/session/{session_id}` - Delete session
- `PATCH /api/chat/session/{session_id}/name` - Rename session
- `POST /api/chat/session/{session_id}/context` - Update session papers
- `POST /api/chat/session/{session_id}/clear` - Clear messages (keep context)
- `DELETE /api/chat/message/{message_id}` - Delete specific message
- `GET /api/chat/history/{session_id}` - Get conversation history
- `DELETE /api/chat/history/{session_id}` - Clear conversation (deprecated)

### Papers

- `GET /api/papers` - List all papers (or filtered by session)
- `GET /api/papers/{paper_id}` - Get paper details
- `POST /api/papers/upload` - Upload PDFs (with optional session_id)
- `POST /api/papers/reindex` - Rebuild vector index from metadata (useful after resets or manual changes)
- `GET /api/papers/{paper_id}/citations` - Get citation graph (OpenAlex)
- `GET /api/files/{filename}` - Serve PDF file content

### Mindmap

- `GET /api/mindmap` - Get knowledge graph
  - Optional params: `session_id`, `paper_id`, `query`
  - Session-scoped: only shows papers from session
  - Validates paper belongs to session if both provided
- `POST /api/mindmap/rebuild` - Regenerate global knowledge graph
- `GET /mindmap` - Interactive D3.js mindmap visualization page

### Tokens

- `GET /api/tokens/usage` - Get total usage
- `GET /api/tokens/usage/{session_id}` - Get session usage
- `GET /api/tokens/history` - Get usage history

### Agent & Config

- `GET /api/agent/profile` - Get agent configuration
- `GET /api/config` - Get app configuration

### Admin

- `POST /api/admin/reset` - Reset all data (dangerous: clears everything)
  - Deletes data/ directory (vector DB, conversations, uploads)
  - Clears in-memory caches (including mindmap caches)
  - Resets paper manager, token tracker, conversation manager
- `POST /api/admin/prune` - Prune unused data
  - Removes papers not referenced by any active session
  - Cleans up:
    - Vector DB chunks
    - Uploaded PDF files
    - Metadata entries
    - OpenAlex citation caches
    - Global mindmap cache

### Health

- `GET /` - Root endpoint (serves frontend)
- `GET /api` - API root
- Health check available via any endpoint

## Configuration

Edit `config.toml` to customize:

- Agent model and parameters
- Token budgets and warnings
- Chunking strategy
- Memory settings

## Capabilities

✅ PDF processing with metadata extraction  
✅ Semantic search using ChromaDB  
✅ RAG pipeline with Google Gemini 2.5 Pro  
✅ **Deep Dive Retrieval**: Automatically scales retrieval density for complex queries  
✅ **Session management with dedicated sidebar**  
✅ **Session duplication (fork context without history)**
✅ **Independent session contexts (papers + chat + tokens)**  
✅ **Messages persist when papers change**  
✅ Conversation history management  
✅ Token usage tracking per session  
✅ **Paper upload via sidebar with auto-indexing**  
✅ **Session-scoped paper context**  
✅ Web-based chat interface  
✅ Collapsible sidebars (papers & sessions)  
✅ Multi-paper query support  
✅ Source citation in responses  
✅ Paper-specific context selection  
✅ **Session-scoped mindmaps with validation**  
✅ **Per-session disk caching for mindmaps**  
✅ AI-generated knowledge graphs (global and paper-specific)  
✅ Interactive D3.js mindmap visualization (NotebookLM-style)  
✅ Canonical titles and concept-focused mindmap structure  
✅ Custom mindmap instructions for alternative views  
✅ Streaming responses with markdown formatting  
✅ **Real-time session updates via SSE events**  

## API Documentation

Interactive API docs available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
