"""Data models and schemas for the application."""

from datetime import datetime
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PaperMetadata(BaseModel):
    """PDF paper metadata."""

    id: str
    filename: str
    filepath: str
    page_count: int
    file_size: int
    created_at: datetime
    indexed_at: Optional[datetime] = None


class PaperChunk(BaseModel):
    """A chunk of text from a paper."""

    id: str
    paper_id: str
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


class Message(BaseModel):
    """Chat message."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    token_count: Optional[int] = None
    source_papers: List[str] = Field(default_factory=list)


class Conversation(BaseModel):
    """Conversation session."""

    session_id: str
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    total_tokens: int = 0


class TokenUsage(BaseModel):
    """Token usage tracking."""

    session_id: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    timestamp: datetime = Field(default_factory=datetime.now)
    model: str


class SessionTokenStats(BaseModel):
    """Token statistics for a session."""

    session_id: str
    total_prompt_tokens: int = 0
    total_response_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str
    session_id: Optional[str] = None
    paper_id: Optional[str] = None  # Optional paper ID to scope chat to specific paper


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    session_id: str
    message: str
    role: MessageRole = MessageRole.ASSISTANT
    source_papers: List[str] = Field(default_factory=list)
    token_usage: Optional[TokenUsage] = None


class PaperListResponse(BaseModel):
    """Response model for papers list endpoint."""

    papers: List[PaperMetadata]
    total_count: int


class AgentProfile(BaseModel):
    """Agent configuration profile."""

    model: str
    max_context_tokens: int
    max_response_tokens: int
    temperature: float
