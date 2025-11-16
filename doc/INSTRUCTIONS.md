# Project Instructions: AI-Powered Journal Article Analysis Web Application

---

## Project Requirements

### Overview

Given a folder with journal articles (PDF), create an AI agent which can answer questions about these papers.

### Application Type

Web application

### GUI Requirements

#### Sidebar (Left, Collapsible)

- Display list of papers from the folder
- Collapsible functionality for space optimization
- Paper selection capability: click to select/unselect a paper
- Visual indication of selected paper (highlight/checkmark)
- Only one paper can be selected at a time

#### Main Window (Chat Interface)

- Function same as ChatGPT app
- Primary area for user-AI agent interaction

### AI Agent Requirements

#### Context Awareness

- Operate over a list of papers in a specified folder
- Access metadata information:
  - File names
  - Number of papers in folder
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

#### Memory Configuration

- Configurable option to remember conversations across sessions
- Session persistence should be toggleable

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
- Forward-focused development approach

#### Guidelines

- Use modern Python features and best practices
- Eliminate unnecessary abstractions
- Write self-documenting code
- Prioritize readability and maintainability
- Remove dead code immediately

---

## Knowledge Graph & Mindmap Visualization

### Goal

After indexing, automatically build a concise knowledge graph representing:

- Key topics per paper
- Relationships between topics and between papers
- Cross-paper connections (e.g., shared methods, datasets, findings)
- **Paper-specific hierarchy**: When a paper is selected, generate a paper-scoped mindmap with that paper as root

### Behavior

- The "📚 Index Papers" action should:
  1) Index PDFs into the vector database, and
  2) Invoke AI to extract topics/relationships and generate a knowledge graph.

- **Paper-scoped mindmap**: When viewing mindmap with a paper selected:
  - Generate a hierarchical tree with the selected paper as the root node
  - Show topics, subtopics, and sub-subtopics covered in that specific paper
  - Structure: Paper Name (root) > Main Topics > Subtopics > Sub-subtopics
  
- **Global mindmap**: When no paper is selected:
  - Show the full cross-paper knowledge graph (all papers organized by themes)

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

### Visualization

- Expose a route `GET /mindmap` that renders an interactive hierarchical tree UI in the browser (NotebookLM-style).
- Use D3.js collapsible tree visualization for clean, organized presentation.
- **Context-aware rendering**:
  - If a paper is selected: show paper-scoped mindmap (paper as root, with its topics/subtopics)
  - If no paper selected: show global cross-paper mindmap
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
