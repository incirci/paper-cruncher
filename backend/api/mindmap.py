"""Mindmap API endpoints."""

from fastapi import APIRouter, Request, Query, HTTPException
from typing import Optional

router = APIRouter()


@router.get("/mindmap")
async def get_mindmap_data(
    request: Request,
    paper_id: Optional[str] = Query(None, description="Optional paper ID to scope mindmap to specific paper"),
    query: Optional[str] = Query(None, description="Optional custom instructions for structuring the mindmap"),
    session_id: Optional[str] = Query(None, description="Optional session ID to scope mindmap to that session's papers"),
):
    """Return the knowledge graph JSON for the mindmap visualization.
    
    If paper_id is provided, returns a tree scoped to that specific paper.
    Otherwise, returns the global tree of all papers.
    """
    app_state = request.app.state.app_state

    if paper_id:
        # Generate a paper-specific mindmap directly from that paper's
        # summary, using its canonical title as the root node. The
        # optional custom query can further shape the structure.
        graph = app_state.mindmap_service.generate_paper_tree(paper_id, custom_query=query)
    elif session_id:
        # Scope the mindmap to the papers attached to the given session.
        conversation = app_state.conversation_manager.get_conversation(session_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Session not found")
        paper_ids = conversation.paper_ids or []
        graph = app_state.mindmap_service.generate_graph(custom_query=query, paper_ids=paper_ids)
    else:
        # For global mindmaps (with or without a custom query), use the
        # generation pipeline across all papers.
        graph = app_state.mindmap_service.generate_graph(custom_query=query)
    
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
