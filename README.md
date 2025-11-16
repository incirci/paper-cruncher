# Journal Article AI Assistant

An AI-powered web application for analyzing and querying journal articles using Google Gemini 2.5 Pro with RAG (Retrieval Augmented Generation).

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Google Gemini API key

### 2. Setup (Local)

```bash
# Install dependencies (already done if you followed initial setup)
pip install -r requirements.txt

# Create .env file with your API key
# (Copy from .env_example and add your key)
```

### 3. Add Papers

Place your PDF journal articles in the `papers/` folder.

### 4. Run the Application

```bash
python run.py
```

The application will start at `http://localhost:8000`

### 5. Open the Web Interface

Open your browser to `http://localhost:8000`

The frontend is automatically served by FastAPI.

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

### Indexing Papers

1. **Add PDFs**: Place PDF journal articles in the `papers/` folder
2. **Index**: Click the "📚 Index Papers" button in the sidebar
   - This processes all PDFs, extracts text, and creates vector embeddings
   - Only needs to be done once, or when adding new papers
   - Papers persist across application restarts

### Asking Questions

1. **Global Mode**: Ask questions about all papers
   - Type your question in the input box at the bottom
   - AI will search across all indexed papers
   - Example: "What are the common themes across these papers?"

2. **Paper-Specific Mode**: Focus on a single paper
   - Click any paper in the sidebar to select it (blue highlight)
   - Ask questions scoped to that paper only
   - Example: "What methodology was used in this study?"
   - Click the paper again to deselect and return to global mode

### Viewing the Knowledge Graph

1. Click the "🗺️ View Mindmap" button in the sidebar
2. **Global Mindmap**: Shows hierarchical tree of topics across all papers
   - Papers organized by themes and subtopics
   - Click nodes to expand/collapse
3. **Paper-Specific Mindmap**: When a paper is selected
   - Shows the selected paper as root
   - Displays main topics and subtopics from that paper
   - Use "Expand All" and "Collapse All" buttons to control view

### Features

- **Conversation History**: Chat maintains context across messages
- **Source Citations**: AI references which papers it's using
- **Streaming Responses**: See answers as they're generated
- **Markdown Formatting**: Responses include tables, lists, and code blocks
- **Token Tracking**: Monitor API usage in real-time

## Project Structure

```text
cruncher/
├── backend/
│   ├── api/              # FastAPI endpoints
│   │   ├── chat.py       # Chat endpoints
│   │   ├── papers.py     # Paper management
│   │   ├── tokens.py     # Token tracking
│   │   ├── agent.py      # Agent config
│   │   └── config.py     # App config
│   ├── core/             # Configuration
│   │   └── config.py     # Settings loader
│   ├── services/         # Business logic
│   │   ├── ai_agent.py           # Gemini integration
│   │   ├── conversation_manager.py
│   │   ├── paper_manager.py
│   │   ├── pdf_processor.py      # PDF extraction
│   │   ├── token_tracker.py      # Token monitoring
│   │   └── vector_db.py          # ChromaDB
│   ├── models/           # Data models
│   │   └── schemas.py    # Pydantic schemas
│   └── main.py           # FastAPI app
├── frontend/
│   └── index.html        # Web UI
├── data/                 # Databases (auto-created)
├── papers/               # PDF storage
├── config.toml           # App configuration
├── .env                  # API keys
└── run.py                # Run script
```

## API Endpoints

### Chat

- `POST /api/chat` - Send message to AI agent
- `GET /api/chat/history/{session_id}` - Get conversation history
- `DELETE /api/chat/history/{session_id}` - Clear conversation
- `GET /api/chat/sessions` - List all sessions

### Papers

- `GET /api/papers` - List all papers
- `GET /api/papers/{paper_id}` - Get paper details
- `POST /api/papers/reindex` - Reindex all papers

### Mindmap

- `GET /api/mindmap` - Get global knowledge graph (all papers)
- `GET /api/mindmap?paper_id=<id>` - Get paper-specific topic tree
- `POST /api/mindmap/rebuild` - Regenerate global knowledge graph
- `GET /mindmap` - Interactive D3.js mindmap visualization

### Tokens

- `GET /api/tokens/usage` - Get total usage
- `GET /api/tokens/usage/{session_id}` - Get session usage
- `GET /api/tokens/history` - Get usage history

### Agent & Config

- `GET /api/agent/profile` - Get agent configuration
- `GET /api/config` - Get app configuration

### Health

- `GET /` - Root endpoint
- `GET /health` - Health check

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
✅ Conversation history management  
✅ Token usage tracking and monitoring  
✅ Web-based chat interface  
✅ Collapsible sidebar with papers list  
✅ Multi-paper query support  
✅ Source citation in responses  
✅ Paper-specific context selection (click to scope chat and mindmap)  
✅ AI-generated knowledge graphs (global and paper-specific)  
✅ Interactive D3.js mindmap visualization (NotebookLM-style)  
✅ Streaming responses with markdown formatting  

## API Documentation

Interactive API docs available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
