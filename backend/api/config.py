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
        },
        "agent": {
            "model": settings.agent.model,
            "max_context_tokens": settings.agent.max_context_tokens,
            "max_response_tokens": settings.agent.max_response_tokens,
            "temperature": settings.agent.temperature,
        },
        "chunking": {
            "chunk_size": settings.chunking.chunk_size,
            "chunk_overlap": settings.chunking.chunk_overlap,
            "max_chunks_per_query": settings.chunking.max_chunks_per_query,
        },
    }
