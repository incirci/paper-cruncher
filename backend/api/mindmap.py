"""Mindmap API endpoints."""

from fastapi import APIRouter, Request, Query
from typing import Optional

router = APIRouter()


@router.get("/mindmap")
async def get_mindmap_data(
    request: Request,
    paper_id: Optional[str] = Query(None, description="Optional paper ID to scope mindmap to specific paper"),
    query: Optional[str] = Query(None, description="Optional custom instructions for structuring the mindmap"),
):
    """Return the knowledge graph JSON for the mindmap visualization.
    
    If paper_id is provided, returns a tree scoped to that specific paper.
    Otherwise, returns the global tree of all papers.
    """
    app_state = request.app.state.app_state
    
    if paper_id:
        # Generate paper-specific tree (custom_query currently not applied;
        # tree is derived from the persisted global mindmap).
        graph = app_state.mindmap_service.generate_paper_tree(paper_id)
    elif query:
        # Custom global mindmap based on user instructions (ephemeral; not persisted)
        graph = app_state.mindmap_service.generate_graph(custom_query=query)
    else:
        # Load global tree (default persisted mindmap)
        graph = app_state.mindmap_service.load_graph()
    
    return graph


@router.post("/mindmap/rebuild")
async def rebuild_mindmap(request: Request):
    """Regenerate the hierarchical knowledge tree from current papers and persist it."""
    app_state = request.app.state.app_state
    tree = app_state.mindmap_service.rebuild_and_persist()
    
    # Count nodes recursively
    def count_nodes(node):
        count = 1
        for child in node.get("children", []):
            count += count_nodes(child)
        return count
    
    return {
        "message": "Mindmap rebuilt successfully",
        "total_nodes": count_nodes(tree),
    }
