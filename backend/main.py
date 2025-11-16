"""Main FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
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


class AppState:
    """Application state container."""

    def __init__(self):
        """Initialize services."""
        # Initialize services
        self.vector_db = VectorDBService(settings.get_vector_db_path())
        self.pdf_processor = PDFProcessor()
        self.paper_manager = PaperManager(self.pdf_processor, self.vector_db)
        self.ai_agent = AIAgent(self.vector_db)
        self.conversation_manager = ConversationManager(
            settings.get_conversation_db_path()
        )
        self.token_tracker = TokenTracker(settings.get_conversation_db_path())
        self.mindmap_service = MindmapService(self.vector_db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    app.state.app_state = AppState()
    app.state.settings = settings  # Make settings accessible to routes
    yield
    # Shutdown (cleanup if needed)


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
      <style>
        body { 
          font-family: system-ui, -apple-system, sans-serif; 
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
        }
        .btn { 
          padding: 8px 16px; 
          border: 1px solid #d0d0d0; 
          background: white;
          border-radius: 6px; 
          cursor: pointer;
          font-size: 14px;
          transition: all 0.2s;
        }
        .btn:hover {
          background: #f5f5f5;
          border-color: #999;
        }
        #status {
          color: #666;
          font-size: 14px;
        }
        #tree-container { 
          width: 100%; 
          height: calc(100vh - 65px);
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
        <span id="status">Loading...</span>
      </div>
      <div id="tree-container"></div>
      
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <script>
        let root, svg, tree, g;
        const width = window.innerWidth;
        const height = window.innerHeight - 65;
        const margin = {top: 20, right: 120, bottom: 20, left: 200};
        
        async function loadTree() {
          document.getElementById('status').textContent = 'Loading...';
          try {
            // Check for paper_id query parameter
            const urlParams = new URLSearchParams(window.location.search);
            const paperId = urlParams.get('paper_id');
            
            // Build API URL with optional paper_id
            let apiUrl = '/api/mindmap';
            if (paperId) {
              apiUrl += `?paper_id=${encodeURIComponent(paperId)}`;
            }
            
            const res = await fetch(apiUrl);
            const data = await res.json();
            renderTree(data);
            document.getElementById('status').textContent = 'Ready';
          } catch (e) {
            console.error(e);
            document.getElementById('status').textContent = 'Failed to load';
          }
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
            .attr('r', 6)
            .attr('class', d => d.children || d._children ? 'has-children' : '');
          
          // Add text with truncation
          const maxChars = 30; // Maximum characters before truncation
          nodeEnter.append('text')
            .attr('dy', '-0.6em')
            .attr('x', d => (d.children || d._children) ? -10 : 10)
            .attr('text-anchor', d => (d.children || d._children) ? 'end' : 'start')
            .text(d => {
              const name = d.data.name;
              return name.length > maxChars ? name.substring(0, maxChars) + '...' : name;
            })
            .append('title') // Add tooltip with full name
            .text(d => d.data.name);
          
          // Update
          const nodeUpdate = nodeEnter.merge(node);
          
          nodeUpdate.transition()
            .duration(duration)
            .attr('transform', d => `translate(${d.y},${d.x})`);
          
          nodeUpdate.select('circle')
            .attr('r', 6)
            .attr('class', d => d.children || d._children ? 'has-children' : '');
          
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
        
        loadTree();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)