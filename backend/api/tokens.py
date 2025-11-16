"""Token usage API endpoints."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/tokens/usage")
async def get_total_usage(request: Request):
    """Get total token usage across all sessions."""
    app_state = request.app.state.app_state
    total_usage = app_state.token_tracker.get_total_usage()
    return total_usage


@router.get("/tokens/usage/{session_id}")
async def get_session_usage(request: Request, session_id: str):
    """Get token usage statistics for a specific session."""
    app_state = request.app.state.app_state
    stats = app_state.token_tracker.get_session_stats(session_id)
    return stats


@router.get("/tokens/history")
async def get_usage_history(request: Request):
    """Get all token usage records."""
    app_state = request.app.state.app_state
    usage_records = app_state.token_tracker.get_all_usage()
    return {"usage_records": usage_records}
