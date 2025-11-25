"""Main FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.api import chat, papers, tokens, agent, config as config_api
from backend.api import mindmap as mindmap_api
from backend.core.config import settings
from backend.services.ai_agent import AIAgent
from backend.services.conversation_manager import ConversationManager
from backend.services.paper_manager import PaperManager
from backend.services.pdf_processor import PDFProcessor
from backend.services.token_tracker import TokenTracker
from backend.services.vector_db import VectorDBService
from backend.services.mindmap_service import MindmapService
from backend.services.citation_service import CitationService
from backend.services.openalex_client import OpenAlexClient
from backend.services.progress_manager import ProgressManager
from fastapi.responses import StreamingResponse


class AppState:
    """Application state container."""

    def __init__(self):
        """Initialize services."""
        # Initialize services
        self.vector_db = VectorDBService(settings.get_vector_db_path())
        self.pdf_processor = PDFProcessor()
        self.openalex_client = OpenAlexClient()
        self.token_tracker = TokenTracker(settings.get_conversation_db_path())
        self.paper_manager = PaperManager(self.pdf_processor, self.vector_db, self.openalex_client)
        self.citation_service = CitationService(self.paper_manager, self.openalex_client)
        self.mindmap_service = MindmapService(self.vector_db)
        self.ai_agent = AIAgent(self.vector_db, self.citation_service)
        self.conversation_manager = ConversationManager(
            settings.get_conversation_db_path()
        )
        self.progress_manager = ProgressManager()



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    app.state.app_state = AppState()
    app.state.settings = settings  # Make settings accessible to routes
    yield
    # Shutdown (cleanup if needed)
    if hasattr(app.state.app_state, "progress_manager"):
        await app.state.app_state.progress_manager.stop()



# Create FastAPI app
app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(papers.router, prefix="/api", tags=["papers"])
app.include_router(tokens.router, prefix="/api", tags=["tokens"])
app.include_router(agent.router, prefix="/api", tags=["agent"])
app.include_router(config_api.router, prefix="/api", tags=["config"])
app.include_router(mindmap_api.router, prefix="/api", tags=["mindmap"])

# Mount uploads directory to serve PDFs
uploads_dir = settings.get_vector_db_path().parent / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/files", StaticFiles(directory=str(uploads_dir)), name="files")


@app.get("/api")
async def root():
    """API root endpoint."""
    return {
        "name": settings.app.name,
        "version": settings.app.version,
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/admin/reset")
async def reset_all_data():
    """Dangerous: wipe the entire data directory on disk.

    This removes *everything* under the configured data root, including:
    - Vector DB (embeddings / Chroma files)
    - Conversation SQLite DB (messages + token usage)
    - Uploaded PDFs
    - Papers metadata JSON
    - Any mindmap/cache artifacts stored under data/

    Intended for development/debugging only.
    """
    from shutil import rmtree

    # The vector DB path lives under the data directory, so use its
    # parent as the canonical data root (e.g. ROOT/data).
    data_root = settings.get_vector_db_path().parent
    app_state: AppState = app.state.app_state

    # 1. Reset services first (logical delete)
    # This clears the DB contents without deleting the locked DB files
    try:
        app_state.vector_db.reset()
    except Exception as e:
        print(f"Error resetting vector DB: {e}")

    try:
        app_state.conversation_manager.reset_all()
    except Exception as e:
        print(f"Error resetting conversation manager: {e}")

    try:
        app_state.token_tracker.reset_all()
    except Exception as e:
        print(f"Error resetting token tracker: {e}")

    try:
        app_state.paper_manager.reset_all()
    except Exception as e:
        print(f"Error resetting paper manager: {e}")

    try:
        app_state.mindmap_service.reset_all()
    except Exception as e:
        print(f"Error resetting mindmap service: {e}")

    # 2. Physically delete non-locked artifacts
    # We explicitly delete folders that contain static files (PDFs, JSONs)
    # but avoid deleting the directories containing active DB files (vectordb, conversations)
    # to prevent "file in use" errors on Windows/Docker.
    
    folders_to_wipe = ["uploads", "cache", "mindmap"]
    for folder in folders_to_wipe:
        path = data_root / folder
        if path.exists():
            try:
                rmtree(path, ignore_errors=True)
                # Recreate empty directory
                path.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Error wiping {folder}: {e}")

    return {"message": f"All data reset: cleared DBs and removed static files in {data_root}"}


@app.post("/api/admin/prune")
async def prune_data():
    """
    Prune data: Delete papers that are not referenced by any active session.
    """
    import json
    app_state: AppState = app.state.app_state
    
    # 1. Get all active sessions
    sessions = app_state.conversation_manager.list_sessions()
    
    # 2. Collect all referenced paper IDs
    referenced_ids = set()
    for session in sessions:
        p_ids_str = session.get("paper_ids")
        if p_ids_str:
            try:
                # It might be a list if the manager was updated, or a string if raw from DB
                if isinstance(p_ids_str, str):
                    p_ids = json.loads(p_ids_str)
                elif isinstance(p_ids_str, list):
                    p_ids = p_ids_str
                else:
                    p_ids = []
                
                if p_ids:
                    referenced_ids.update(p_ids)
            except Exception as e:
                print(f"Error parsing paper_ids for session {session.get('session_id')}: {e}")
    
    # 3. Prune papers
    removed_count = app_state.paper_manager.prune_papers(list(referenced_ids))
    
    # 4. Invalidate mindmap cache if papers were removed
    if removed_count > 0:
        app_state.mindmap_service.invalidate_global_cache()
    
    return {
        "message": f"Pruned {removed_count} papers that were not referenced by any session.",
        "removed_count": removed_count
    }


# Serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML."""
        return FileResponse(frontend_path / "index.html")
    
    # Mount static files if needed
    # app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Simple server-rendered mindmap page with D3.js collapsible tree
@app.get("/mindmap")
async def serve_mindmap_page():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Knowledge Map</title>
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
      <style>
        body { 
          font-family: 'Inter', system-ui, -apple-system, sans-serif; 
          margin: 0; 
          padding: 0;
          background: #fafafa;
        }
        #toolbar { 
          padding: 16px 20px; 
          background: white;
          border-bottom: 1px solid #e0e0e0; 
          display: flex; 
          gap: 12px; 
          align-items: center;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
          flex-wrap: wrap;
        }
        .filter-group {
            display: flex;
            align-items: center;
            gap: 8px;
            border-right: 1px solid #e0e0e0;
            padding-right: 12px;
        }
        .filter-group:last-child {
            border-right: none;
        }
        .filter-group label {
            font-size: 12px;
            font-weight: 600;
            color: #555;
        }
        .year-input {
            width: 50px;
            padding: 4px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 12px;
        }
        .btn { 
          padding: 8px 16px; 
          border: 1px solid #d1d5db; 
          background: white;
          border-radius: 6px; 
          cursor: pointer;
          font-size: 13px;
          transition: all 0.2s;
          color: #374151;
          font-weight: 500;
        }
        .btn:hover {
          background: #f3f4f6;
          border-color: #9ca3af;
        }
        .btn-primary {
            background: #e8f0fe;
            color: #1967d2;
            border-color: #d2e3fc;
        }
        .btn-primary:hover {
            background: #d2e3fc;
            color: #185abc;
            border-color: #c2d7fa;
        }
        #status {
          color: #666;
          font-size: 14px;
          margin-left: auto;
        }
        #tree-container { 
          width: 100%; 
          height: calc(100vh - 80px);
          overflow: auto;
        }
        .node circle {
          fill: #fff;
          stroke: #4a90e2;
          stroke-width: 2px;
          cursor: pointer;
        }
        .node circle.has-children {
          fill: #4a90e2;
        }
        .node circle.is-local {
          fill: #2ecc71; /* Green for local papers */
          stroke: #27ae60;
        }
        .node text {
          font-size: 13px;
          font-family: system-ui, sans-serif;
          cursor: pointer;
          fill: #333;
        }
        .link {
          fill: none;
          stroke: #ccc;
          stroke-width: 2px;
        }
        .node:hover circle {
          stroke: #2563eb;
          stroke-width: 3px;
        }
      </style>
    </head>
    <body>
      <div id="toolbar">
        <button class="btn" id="refreshBtn">🔄 Refresh</button>
        <button class="btn" id="expandAllBtn">⊕ Expand All</button>
        <button class="btn" id="collapseAllBtn">⊖ Collapse All</button>
        
        <div id="citationFilters" style="display: none; gap: 12px; align-items: center;">
            <div class="filter-group">
                <label>Keyword:</label>
                <input type="text" id="keywordFilter" class="year-input" style="width: 100px;" placeholder="e.g. *neural*">
            </div>

            <div class="filter-group">
                <label>Refs Year:</label>
                <input type="number" id="refStart" class="year-input" placeholder="Min">
                <span>-</span>
                <input type="number" id="refEnd" class="year-input" placeholder="Max">
            </div>

            <div class="filter-group">
                <label>Cites Year:</label>
                <input type="number" id="citeStart" class="year-input" placeholder="Min">
                <span>-</span>
                <input type="number" id="citeEnd" class="year-input" placeholder="Max">
            </div>

            <button class="btn btn-primary" id="applyFiltersBtn">Filter</button>
        </div>

        <span id="status">Loading...</span>
      </div>
      <div id="tree-container"></div>
      
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <script>
        let root, svg, tree, g;
        let originalData = null;
        let globalMinCitations = 0;
        let globalMaxCitations = 0;
        const width = window.innerWidth;
        const height = window.innerHeight - 80;
        const margin = {top: 20, right: 120, bottom: 20, left: 200};

        function getRadius(d) {
             if (d.data.citation_count === undefined) return 6;
             
             const count = d.data.citation_count;
             const minR = __MIN_R__;
             const maxR = __MAX_R__;
             
             if (globalMaxCitations <= globalMinCitations) return minR;
             
             const minLog = Math.log(globalMinCitations + 1);
             const maxLog = Math.log(globalMaxCitations + 1);
             const valLog = Math.log(count + 1);
             
             // Linear interpolation on log scale
             const t = (valLog - minLog) / (maxLog - minLog);
             return minR + t * (maxR - minR);
        }
        
        async function loadTree() {
          document.getElementById('status').textContent = 'Loading...';
          try {
            // Check for query parameters
            const urlParams = new URLSearchParams(window.location.search);
            const paperId = urlParams.get('paper_id');
            const query = urlParams.get('query');
            const sessionId = urlParams.get('session_id');
            const mode = urlParams.get('mode'); // 'mindmap' (default) or 'citations'
            
            let apiUrl;
            
            if (mode === 'citations' && paperId) {
                // Citation graph mode
                apiUrl = `/api/papers/${paperId}/citations`;
                document.title = "Citation Map";
                document.getElementById('citationFilters').style.display = 'flex';
            } else {
                // Standard mindmap mode
                const apiParams = new URLSearchParams();
                if (sessionId) apiParams.set('session_id', sessionId);
                if (paperId) apiParams.set('paper_id', paperId);
                if (query) apiParams.set('query', query);
                apiUrl = apiParams.toString() ? `/api/mindmap?${apiParams.toString()}` : '/api/mindmap';
                document.getElementById('citationFilters').style.display = 'none';
            }
            
            const res = await fetch(apiUrl);
            originalData = await res.json();
            console.log("Loaded data:", originalData);
            
            // Calculate year ranges from data
            let refMin = 9999, refMax = 0;
            let citeMin = 9999, citeMax = 0;
            globalMinCitations = 999999;
            globalMaxCitations = 0;

            function findYears(node, context) {
                // Track citations
                if (node.citation_count !== undefined) {
                    if (node.citation_count < globalMinCitations) globalMinCitations = node.citation_count;
                    if (node.citation_count > globalMaxCitations) globalMaxCitations = node.citation_count;
                }

                let currentContext = context;
                if (node.name && node.name.startsWith("References")) currentContext = 'ref';
                if (node.name && node.name.startsWith("Cited By")) currentContext = 'cite';

                if (node.children && node.children.length > 0) {
                    for (const child of node.children) findYears(child, currentContext);
                } else if (node.year) {
                    const y = parseInt(node.year);
                    if (!isNaN(y)) {
                        if (currentContext === 'ref') {
                            if (y < refMin) refMin = y;
                            if (y > refMax) refMax = y;
                        } else if (currentContext === 'cite') {
                            if (y < citeMin) citeMin = y;
                            if (y > citeMax) citeMax = y;
                        }
                    }
                }
            }
            
            findYears(originalData, 'root');
            
            if (globalMinCitations === 999999) globalMinCitations = 0;
            console.log("Citations range:", globalMinCitations, globalMaxCitations);

            const currentYear = new Date().getFullYear();
            if (refMin === 9999) refMin = 1900;
            if (refMax === 0) refMax = currentYear;
            if (citeMin === 9999) citeMin = 1900;
            if (citeMax === 0) citeMax = currentYear;

            document.getElementById('refStart').value = refMin;
            document.getElementById('refEnd').value = refMax;
            document.getElementById('citeStart').value = citeMin;
            document.getElementById('citeEnd').value = citeMax;

            applyFilters();
            document.getElementById('status').textContent = 'Ready';
          } catch (e) {
            console.error(e);
            document.getElementById('status').textContent = 'Failed to load';
          }
        }

        function applyFilters() {
            if (!originalData) return;

            const refStartInput = document.getElementById('refStart').value;
            const refEndInput = document.getElementById('refEnd').value;
            const citeStartInput = document.getElementById('citeStart').value;
            const citeEndInput = document.getElementById('citeEnd').value;
            const keywordInputEl = document.getElementById('keywordFilter');
            const keywordInput = keywordInputEl ? keywordInputEl.value.trim() : '';

            const refStart = refStartInput ? parseInt(refStartInput) : 0;
            const refEnd = refEndInput ? parseInt(refEndInput) : 9999;
            const citeStart = citeStartInput ? parseInt(citeStartInput) : 0;
            const citeEnd = citeEndInput ? parseInt(citeEndInput) : 9999;

            let keywordRegex = null;
            if (keywordInput) {
                try {
                    // Escape special regex chars: . + ? ^ $ { } ( ) | [ ] \
                    // We do NOT escape * because we use it as wildcard
                    const escapeRegex = (str) => {
                        return str.replace(/[.+?^${}()|[\\]\\\\]/g, (match) => {
                            return '\\\\' + match;
                        });
                    };
                    
                    // Split by *, escape parts, join by .*
                    const parts = keywordInput.split('*');
                    const escapedParts = parts.map(escapeRegex);
                    const pattern = escapedParts.join('.*');
                    
                    keywordRegex = new RegExp(pattern, 'i');
                } catch (e) {
                    console.error("Regex error", e);
                }
            }

            console.log(`Filtering: Refs ${refStart}-${refEnd}, Cites ${citeStart}-${citeEnd}, Keyword: ${keywordInput}`);

            // Deep copy to avoid mutating originalData
            const data = JSON.parse(JSON.stringify(originalData));

            function countLeaves(node) {
                if (!node.children || node.children.length === 0) return 1;
                return node.children.reduce((sum, child) => sum + countLeaves(child), 0);
            }

            function filterNode(node, context) {
                // context: 'root', 'ref', 'cite'
                let currentContext = context;
                
                if (node.name && node.name.startsWith("References")) currentContext = 'ref';
                if (node.name && node.name.startsWith("Cited By")) currentContext = 'cite';

                if (node.children && node.children.length > 0) {
                    // Group node (Section, Topic, Concept)
                    const filteredChildren = [];
                    for (const child of node.children) {
                        if (filterNode(child, currentContext)) {
                            filteredChildren.push(child);
                        }
                    }
                    node.children = filteredChildren;
                    
                    // Update label count if present
                    if (node.name && node.name.match(/\\(\\d+\\)$/)) {
                        const count = countLeaves(node);
                        node.name = node.name.replace(/\\(\\d+\\)$/, `(${count})`);
                    }
                    
                    // Update citation_count for the group node based on remaining children
                    if (node.citation_count !== undefined) {
                         node.citation_count = node.children.reduce((sum, child) => sum + (child.citation_count || 0), 0);
                    }

                    // Sort children by citation_count descending
                    node.children.sort((a, b) => (b.citation_count || 0) - (a.citation_count || 0));
                    
                    // Keep this node if it still has children
                    return node.children.length > 0;
                } else {
                    // Leaf node (Paper)
                    if (currentContext === 'root') return true; 

                    // Check keyword if present
                    if (keywordRegex) {
                        if (!node.name || typeof node.name !== 'string') return false;
                        if (!keywordRegex.test(node.name)) return false;
                    }

                    // If no year, keep it (or could filter out if strict)
                    if (!node.year) return true;
                    
                    const y = parseInt(node.year);
                    if (currentContext === 'ref') {
                        return y >= refStart && y <= refEnd;
                    }
                    if (currentContext === 'cite') {
                        return y >= citeStart && y <= citeEnd;
                    }
                    return true;
                }
            }

            // Filter the root's children
            if (data.children) {
                const validSections = [];
                for (const section of data.children) {
                    if (filterNode(section, 'root')) {
                        validSections.push(section);
                    }
                }
                data.children = validSections;
            }

            // Recalculate citation range for the filtered data
            globalMinCitations = 999999;
            globalMaxCitations = 0;
            
            function findMinMax(node) {
                if (node.citation_count !== undefined) {
                    if (node.citation_count < globalMinCitations) globalMinCitations = node.citation_count;
                    if (node.citation_count > globalMaxCitations) globalMaxCitations = node.citation_count;
                }
                if (node.children) {
                    node.children.forEach(findMinMax);
                }
            }
            findMinMax(data);
            if (globalMinCitations === 999999) globalMinCitations = 0;
            console.log("Recalculated citations range:", globalMinCitations, globalMaxCitations);

            renderTree(data);
        }
        
        function renderTree(data) {
          // Clear previous
          d3.select('#tree-container').selectAll('*').remove();
          
          // Create SVG
          svg = d3.select('#tree-container')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
          
          g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
          
          // Create tree layout
          tree = d3.tree().size([height - margin.top - margin.bottom, width - margin.left - margin.right - 200]);
          
          // Create hierarchy
          root = d3.hierarchy(data);
          root.x0 = (height - margin.top - margin.bottom) / 2;
          root.y0 = 0;
          
          // Collapse all children initially except first level
          if (root.children) {
            root.children.forEach(collapse);
          }
          
          update(root);
        }
        
        function collapse(d) {
          if (d.children) {
            d._children = d.children;
            d._children.forEach(collapse);
            d.children = null;
          }
        }
        
        function expand(d) {
          if (d._children) {
            d.children = d._children;
            d._children = null;
            if (d.children) d.children.forEach(expand);
          }
        }
        
        function update(source) {
          const duration = 300;
          const treeData = tree(root);
          const nodes = treeData.descendants();
          const links = treeData.links();
          
          // Normalize for fixed-depth (increase spacing between levels)
          nodes.forEach(d => { d.y = d.depth * 250; });
          
          // Update nodes
          const node = g.selectAll('g.node')
            .data(nodes, d => d.id || (d.id = ++i));
          
          // Enter new nodes
          const nodeEnter = node.enter().append('g')
            .attr('class', 'node')
            .attr('transform', d => `translate(${source.y0},${source.x0})`)
            .on('click', click);
          
          nodeEnter.append('circle')
            .attr('r', d => getRadius(d))
            .attr('class', d => {
                if (d.data.is_local) return 'is-local';
                return d.children || d._children ? 'has-children' : '';
            });
          
          // Add text with conditional truncation
          const maxChars = 30;
          nodeEnter.append('text')
            .attr('dy', '-0.6em')
            .attr('x', d => (d.children || d._children) ? -10 : 10)
            .attr('text-anchor', d => (d.children || d._children) ? 'end' : 'start')
            .text(d => {
              const name = d.data.name;
              const isIntermediate = d.children || d._children;
              return (isIntermediate && name.length > maxChars) 
                ? name.substring(0, maxChars) + '...' 
                : name;
            })
            .append('title')
            .text(d => {
                let txt = d.data.name;
                if (d.data.citation_count !== undefined) {
                    txt += `\nCitations: ${d.data.citation_count}`;
                }
                return txt;
            });
          
          // Update
          const nodeUpdate = nodeEnter.merge(node);
          
          nodeUpdate.transition()
            .duration(duration)
            .attr('transform', d => `translate(${d.y},${d.x})`);
          
          nodeUpdate.select('circle')
            .attr('r', d => getRadius(d))
            .attr('class', d => {
                if (d.data.is_local) return 'is-local';
                return d.children || d._children ? 'has-children' : '';
            });
          
          // Exit
          const nodeExit = node.exit().transition()
            .duration(duration)
            .attr('transform', d => `translate(${source.y},${source.x})`)
            .remove();
          
          nodeExit.select('circle').attr('r', 0);
          nodeExit.select('text').style('fill-opacity', 0);
          
          // Update links
          const link = g.selectAll('path.link')
            .data(links, d => d.target.id);
          
          const linkEnter = link.enter().insert('path', 'g')
            .attr('class', 'link')
            .attr('d', d => {
              const o = {x: source.x0, y: source.y0};
              return diagonal(o, o);
            });
          
          const linkUpdate = linkEnter.merge(link);
          
          linkUpdate.transition()
            .duration(duration)
            .attr('d', d => diagonal(d.source, d.target));
          
          link.exit().transition()
            .duration(duration)
            .attr('d', d => {
              const o = {x: source.x, y: source.y};
              return diagonal(o, o);
            })
            .remove();
          
          // Store old positions
          nodes.forEach(d => {
            d.x0 = d.x;
            d.y0 = d.y;
          });
        }
        
        function diagonal(s, d) {
          return `M ${s.y} ${s.x}
                  C ${(s.y + d.y) / 2} ${s.x},
                    ${(s.y + d.y) / 2} ${d.x},
                    ${d.y} ${d.x}`;
        }
        
        let i = 0;
        
        function click(event, d) {
          // If it's a leaf node with a URL, open it
          if (d.data.url && !d.children && !d._children) {
            window.open(d.data.url, '_blank');
            return;
          }

          if (d.children) {
            d._children = d.children;
            d.children = null;
          } else {
            d.children = d._children;
            d._children = null;
          }
          update(d);
        }
        
        function expandAll() {
          if (!root) return;

          function expandRecursive(node) {
            if (node._children) {
              node.children = node._children;
              node._children = null;
            }
            (node.children || []).forEach(expandRecursive);
          }

          expandRecursive(root);
          update(root);
        }

        function collapseAll() {
          if (!root) return;

          function collapseRecursive(node) {
            if (node.children) {
              node._children = node.children;
              node.children = null;
            }
            (node._children || []).forEach(collapseRecursive);
          }

          collapseRecursive(root);
          update(root);
        }
        
        document.getElementById('refreshBtn').addEventListener('click', loadTree);
        document.getElementById('expandAllBtn').addEventListener('click', expandAll);
        document.getElementById('collapseAllBtn').addEventListener('click', collapseAll);
        document.getElementById('applyFiltersBtn').addEventListener('click', applyFilters);
        
        loadTree();
      </script>
    </body>
    </html>
    """
    html = html.replace("__MIN_R__", str(settings.mindmap.citation_node_min_size))
    html = html.replace("__MAX_R__", str(settings.mindmap.citation_node_max_size))
    return HTMLResponse(content=html)


@app.get("/api/progress")
async def progress_stream():
    """Stream progress updates via SSE."""
    return StreamingResponse(
        app.state.app_state.progress_manager.listen(),
        media_type="text/event-stream"
    )