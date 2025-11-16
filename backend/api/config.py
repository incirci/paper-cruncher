"""Configuration API endpoints."""

from fastapi import APIRouter, Request

from backend.core.config import settings

router = APIRouter()


@router.get("/config")
async def get_config(request: Request):
    """Get application configuration (non-sensitive values)."""
    return {
        "app": {
            "name": settings.app.name,
            "version": settings.app.version,
            "papers_folder": settings.app.papers_folder,
        },
        "agent": {
            "model": settings.agent.model,
            "max_context_tokens": settings.agent.max_context_tokens,
            "max_response_tokens": settings.agent.max_response_tokens,
            "temperature": settings.agent.temperature,
        },
        "memory": {
            "persist_across_sessions": settings.memory.persist_across_sessions,
            "max_history_messages": settings.memory.max_history_messages,
            "enable_summarization": settings.memory.enable_summarization,
        },
        "tokens": {
            "track_usage": settings.tokens.track_usage,
            "session_budget": settings.tokens.session_budget,
            "enable_warnings": settings.tokens.enable_warnings,
        },
        "chunking": {
            "chunk_size": settings.chunking.chunk_size,
            "chunk_overlap": settings.chunking.chunk_overlap,
            "max_chunks_per_query": settings.chunking.max_chunks_per_query,
        },
    }
