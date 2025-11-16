"""Agent configuration API endpoints."""

from fastapi import APIRouter, Request

from backend.core.config import settings
from backend.models.schemas import AgentProfile

router = APIRouter()


@router.get("/agent/profile", response_model=AgentProfile)
async def get_agent_profile(request: Request):
    """Get current agent configuration profile."""
    return AgentProfile(
        model=settings.agent.model,
        max_context_tokens=settings.agent.max_context_tokens,
        max_response_tokens=settings.agent.max_response_tokens,
        temperature=settings.agent.temperature,
    )
