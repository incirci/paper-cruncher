"""Papers API endpoints."""

from fastapi import APIRouter, HTTPException, Request

from backend.models.schemas import PaperListResponse

router = APIRouter()


@router.get("/papers", response_model=PaperListResponse)
async def list_papers(request: Request):
    """
    Get list of all papers in the collection.

    Returns:
        List of papers with metadata
    """
    app_state = request.app.state.app_state
    papers = app_state.paper_manager.list_papers()

    return PaperListResponse(
        papers=papers,
        total_count=len(papers),
    )


@router.get("/papers/{paper_id}")
async def get_paper(request: Request, paper_id: str):
    """
    Get details of a specific paper.

    Args:
        paper_id: Paper ID

    Returns:
        Paper metadata
    """
    app_state = request.app.state.app_state
    paper = app_state.paper_manager.get_paper(paper_id)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    return paper


@router.post("/papers/reindex")
async def reindex_papers(request: Request):
    """Reindex all papers in the papers folder."""
    app_state = request.app.state.app_state
    app_state.paper_manager.reindex_all()

    papers = app_state.paper_manager.list_papers()

    # After indexing, rebuild the mindmap for updated knowledge graph
    try:
        tree = app_state.mindmap_service.rebuild_and_persist()
        
        def count_nodes(node):
            count = 1
            for child in node.get("children", []):
                count += count_nodes(child)
            return count
        
        mindmap_stats = {
            "total_nodes": count_nodes(tree),
            "path": str(app_state.mindmap_service.graph_file),
        }
    except Exception as e:
        mindmap_stats = {"error": str(e)}

    return {
        "message": "Papers reindexed successfully",
        "total_papers": len(papers),
        "mindmap": mindmap_stats,
    }
