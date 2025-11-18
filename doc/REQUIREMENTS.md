# Project Instructions: AI-Powered Journal Article Analysis Web Application

---

## Project Requirements

### Overview

Given a set of journal articles (PDF), create an AI agent which can answer questions about these papers. Papers are added via upload/drop-in in the UI.

### Application Type

Web application

### GUI Requirements

#### Sidebar (Left, Collapsible)

- Display list of papers available in the current session (uploaded/side-loaded PDFs)
- Collapsible functionality for space optimization
- Paper selection capability: click to select/unselect a paper
- Visual indication of selected paper (highlight/checkmark)
- Only one paper can be selected at a time
- Support **drop-in / file-select upload** of papers (PDFs) via the sidebar as the primary and only way to add content (no legacy `papers/` folder scanning in the UI)
  - Uploaded papers are saved under a persistent `uploads/` directory and indexed into the vector database for reuse across sessions, as long as the underlying files remain
  - Uploaded papers are clearly labeled in the sidebar (e.g., "uploaded" badge or section) and displayed using their **canonical title** (e.g. `filename (inferred_title)` when helpful)
  - Re-uploading a PDF that is already indexed (same file path under `uploads/`) must be fast: the system should reuse existing metadata and embeddings instead of re-extracting text and re-indexing chunks.

#### Main Window (Chat Interface)

- Function same as ChatGPT app
- Primary area for user-AI agent interaction
- Support a **scroll-to-bottom** affordance when the user scrolls up in a long conversation so it is easy to jump back to the newest answer

#### Chat Session Management

- Provide a **New Chat** control to start a fresh conversation
- A new chat should:
  - Clear the visible message history in the UI
  - Start a new backend session (new `session_id`) so token usage and context are tracked separately
  - Keep the currently selected paper context (if any) unless the user explicitly changes it
  - When the session is saved, also persist any **uploaded/side-loaded papers** that were added during that session so the same logical session (chat history + paper set) can be reloaded later.

### AI Agent Requirements

#### Context Awareness

- Operate over a list of papers available in the current session (uploaded during the session and/or previously indexed uploads that still exist on disk)
- Access metadata information:
  - File names
  - Number of papers available in the current indexed set
  - Other relevant file metadata
- Answer questions about file contents

#### Capabilities

- Find useful content across multiple papers
- Provide paper-specific answers
- Provide consolidated answers across papers
- Remember communication history with user
- **Context-aware operation**: When a paper is selected in the sidebar:
  - All chat questions are answered specifically for that paper only
  - Questions are scoped to the selected paper's content
  - Context is clearly indicated to the user
- **Default behavior**: When no paper is selected, operate across all papers as before

#### Message-Level UX

- For AI responses that contain code blocks, provide **copy buttons on each code block** in addition to the whole-message copy control
- Code-block copy should copy exactly the contents of that block (without extra prompt/metadata text)

#### File and Image Upload

- Add support for **file/image upload** directly from the chat interface
- Uploaded files should, at a minimum, be available to the backend as attachments for the current session (e.g., for ad‑hoc inspection or future indexing)
- For now, uploading **PDFs** or other supported document types does not need to automatically trigger indexing, but the architecture should allow this in a later phase

#### Memory Configuration

- Configurable option to remember conversations across sessions
- Session persistence should be toggleable
- When enabled, session persistence should include:
  - Chat messages
  - Selected paper (if any)
  - References to any uploaded papers that were indexed for that session, so that reloading a session restores both context and available papers.

### Technical Configuration

#### Configuration Files

- **config.toml**: Application configuration parameters
- **.env**: Environment parameters (API tokens, secrets)
- **.env_example**: Template file for environment variables (to be created)

#### Documentation

- Minimal documentation effort required
- No extensive README needed

#### Testing Strategy

- Application should be ready for unit test integration
- Define endpoints accordingly for testability
- **Do not create unit tests yet** (infrastructure only)

#### Technology Stack

- **Primary Language**: Python (use as much as possible)
- **AI Model**: Google Gemini 2.5 Pro

### AI Agent Profiles

- Create agent profile settings per agent
- Currently support: Google Gemini 2.5 Pro only
- Extensible architecture for future agent additions

### Token Management

#### Monitoring Requirements

- Track token usage during active session
- Track token usage across sessions (historical data)

#### Optimization Requirements

- Create efficient architecture to optimize token usage
- Implement smart chunking and retrieval strategies
- Use caching where appropriate
- Minimize redundant API calls

### Code Quality Standards

#### Principles

- Keep code simple and concise
- Well-structured architecture
- Avoid redundancy
- No fallbacks for legacy compatibility
- No backward compatibility concerns
- Clean, modern code only

#### Guidelines

- Use modern Python features and best practices
- Eliminate unnecessary abstractions
- Prioritize readability and maintainability
- Remove dead code immediately

### Behavior

The system should automatically index uploaded PDFs when they are added via the sidebar (no separate "Index Papers" button in the UI). Internally, the indexing pipeline is:

Internally, the pipeline is:

1. PDFs are processed into text chunks and `PaperMetadata` records.
2. Each paper gets a **canonical title** (e.g. `filename (inferred_title)` when helpful) used consistently in the sidebar, vector DB, and mindmap.
3. `VectorDBService` stores chunks in ChromaDB and builds a short, keyword-biased **micro-summary** per paper from representative chunks.
4. `MindmapService` calls the LLM with a concept-focused prompt that:
   - Uses canonical titles for leaves.
   - Asks for internal node names that describe conceptual content (what/why/how/context), not section names or publication types.
5. The raw JSON tree from the LLM is post-processed to normalize and de-duplicate internal concept nodes before being persisted.

- **Semantic hierarchy requirement**: In both global and paper-scoped mindmaps, each child node must be a semantic refinement of its parent (no generic structural labels), and sibling nodes under the same parent must share a clear, coherent theme.

- **Paper-scoped mindmap**: When viewing mindmap with a paper selected:
  - Generate a hierarchical tree with the selected paper as the root node
  - Show topics, subtopics, and sub-subtopics covered in that specific paper
  - Structure: Paper Name (root) > Main Topics > Subtopics > Sub-subtopics
  
- **Global mindmap**: When no paper is selected:
  - Show the full cross-paper knowledge graph (all papers organized by themes)
  - Support user-provided **mindmap queries** that influence how the hierarchy is organized while still respecting all structural constraints (valid JSON, canonical titles as leaf nodes, depth limits, semantic parent–child relationships)

### Persistence

- Save the graph as JSON under `data/mindmap/graph.json` for fast retrieval.
- Structure: hierarchical tree format for D3.js visualization

  ```json
  {
    "name": "Root",
    "children": [
      {
        "name": "Topic",
        "children": [{"name": "Paper"}]
      }
    ]
  }
  ```

- **Paper-specific mindmaps**: Generate on-demand for selected paper, not persisted (ephemeral)
  - Cache generation results in memory keyed by `(paper_id, query)` to avoid redundant LLM calls for the same paper and instructions within a run

- **Global mindmap caching**:
  - Cache and reuse global mindmaps based on the current set of indexed papers and the active mindmap query
  - Persist the default (no custom query) global mindmap to `data/mindmap/graph.json` and track fingerprints in an index file so matching configurations can reuse the existing graph across restarts

### Visualization

- Expose a route `GET /mindmap` that renders an interactive hierarchical tree UI in the browser (NotebookLM-style).
- Use D3.js collapsible tree visualization for clean, organized presentation.
- **Context-aware rendering**:
  - If a paper is selected: show paper-scoped mindmap (paper as root, with its topics/subtopics)
  - If no paper selected: show global cross-paper mindmap
    - If a custom mindmap query is provided by the user, use it to influence how the hierarchy is organized while still respecting all structural constraints (valid JSON, canonical titles as leaf nodes, depth limits).
- The UI loads hierarchical data from `GET /api/mindmap?paper_id=<id>` (optional param for paper-scoped view) in nested JSON format:

  ```json
  {
    "name": "Root",
    "children": [
      {
        "name": "Category/Topic",
        "children": [
          {"name": "Paper 1"},
          {"name": "Paper 2"}
        ]
      }
    ]
  }
  ```

- Support collapsible nodes (click to expand/collapse branches).
- No additional client build tooling is required; use CDN-based D3.js.

### Quality Bar

- Hierarchy must be deterministic and concise (limit topics to essentials).
- Prefer stable, low-temperature generation for the structure extraction step.
- Handle empty/no-paper states gracefully.
- Nodes should be collapsible for better navigation of complex structures.
- Internal node names should be concept-focused and non-generic, while paper leaves must use the canonical titles from the papers list.
