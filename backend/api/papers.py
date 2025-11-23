"""Papers API endpoints."""

from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form

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
    """Reindex all known papers from stored metadata.

    This rebuilds the vector DB and then refreshes the mindmap,
    keeping uploaded papers intact.
    """
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


@router.post("/papers/upload", response_model=PaperListResponse)
async def upload_papers(
    request: Request,
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(None),
):
    """Upload one or more PDF papers and index them.

    Files are stored under an uploads directory and immediately indexed
    so they become available in the current session.
    """

    app_state = request.app.state.app_state
    paper_manager = app_state.paper_manager

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    uploaded_papers = []

    for upload in files:
        original_name = upload.filename or ""
        if not original_name.lower().endswith(".pdf"):
            # Skip non-PDFs quietly for now
            continue

        paper_manager.uploads_dir.mkdir(parents=True, exist_ok=True)
        # Store on disk under the uploads directory. For now we keep the
        # original filename on disk as well; collisions are unlikely in
        # typical use and can be handled later if needed.
        target_path = paper_manager.uploads_dir / original_name

        try:
            with target_path.open("wb") as f:
                content = await upload.read()
                f.write(content)

            # Index the paper and collect the returned metadata so we can
            # return only the newly uploaded papers to the caller.
            metadata = paper_manager.add_paper(target_path, original_filename=original_name)
            uploaded_papers.append(metadata)
        except Exception as exc:  # pragma: no cover - logged, but non-fatal
            # Log error to console and continue with remaining files
            print(f"Error processing uploaded paper {original_name}: {exc}")

    # If a session_id is provided, attach uploaded papers to that session
    # and clear its chat history/token usage if the paper set changed.
    if session_id:
        try:
            conv_mgr = app_state.conversation_manager
            tok = app_state.token_tracker
            if conv_mgr.session_exists(session_id):
                conv = conv_mgr.get_conversation(session_id)
                old_ids = set((conv.paper_ids if conv else []) or [])
                new_ids = list(old_ids.union({p.id for p in uploaded_papers}))
                if set(new_ids) != old_ids:
                    conv_mgr.update_session_papers(
                        session_id=session_id,
                        selected_paper_id=(conv.selected_paper_id if conv else None),
                        paper_ids=new_ids,
                    )
        except Exception as e:  # pragma: no cover - defensive
            # Do not fail the upload if session update fails; just log.
            print(f"Warning: failed to attach uploaded papers to session {session_id}: {e}")

    return PaperListResponse(
        papers=uploaded_papers,
        total_count=len(uploaded_papers),
    )


@router.get("/papers/{paper_id}/citations")
async def get_paper_citations(request: Request, paper_id: str):
    """
    Get citation graph for a specific paper using Semantic Scholar.
    
    Returns a D3.js compatible tree structure with:
    - References (backward citations)
    - Citations (forward citations)
    - Local paper resolution (is_local=True if we have the PDF)
    """
    app_state = request.app.state.app_state
    
    # Check if paper exists locally first
    paper = app_state.paper_manager.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
        
    return await app_state.citation_service.get_citation_graph(paper_id)
