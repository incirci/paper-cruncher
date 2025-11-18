"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Optional

import toml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent


class AppConfig(BaseModel):
    """Application configuration."""

    name: str = "Journal Article AI Assistant"
    version: str = "1.0.0"
    papers_folder: str = "./papers"
    host: str = "0.0.0.0"
    port: int = 8000


class AgentConfig(BaseModel):
    """AI Agent configuration."""

    model: str = "gemini-2.0-flash-exp"
    max_context_tokens: int = 1000000
    max_response_tokens: int = 8192
    temperature: float = 0.7
    # Optional separate model/config for orchestrator step
    orchestrator_model: Optional[str] = None
    orchestrator_temperature: float = 0.2
    orchestrator_max_output_tokens: int = 300


class MemoryConfig(BaseModel):
    """Conversation memory configuration."""

    persist_across_sessions: bool = True
    max_history_messages: int = 50
    enable_summarization: bool = True
    summarization_threshold: int = 20


class TokensConfig(BaseModel):
    """Token tracking configuration."""

    track_usage: bool = True
    session_budget: int = 1000000
    enable_warnings: bool = True
    warning_threshold: float = 0.8


class DatabaseConfig(BaseModel):
    """Database configuration."""

    vector_db_path: str = "./data/vectordb"
    conversation_db_path: str = "./data/conversations.db"


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_query: int = 5
    title_snippet_chars: int = 1000


class MindmapConfig(BaseModel):
    """Mindmap generation configuration."""

    max_depth: int = 4
    min_themes: int = 3
    max_themes: int = 7
    node_name_max_length: int = 60


class ImageConfig(BaseModel):
    """Image generation configuration (Imagen)."""

    enabled: bool = False
    model: str = "imagegeneration"
    mime_type: str = "image/png"
    width: int = 1024
    height: int = 1024


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Settings(BaseModel):
    """Application settings loaded from config.toml and environment variables."""

    app: AppConfig = Field(default_factory=AppConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tokens: TokensConfig = Field(default_factory=TokensConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    mindmap: MindmapConfig = Field(default_factory=MindmapConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Environment variables
    google_api_key: str = Field(default="")

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from config.toml and environment variables."""
        if config_path is None:
            config_path = ROOT_DIR / "config.toml"

        # Load TOML configuration
        if config_path.exists():
            config_data = toml.load(config_path)
        else:
            config_data = {}

        # Create settings instance
        settings = cls(
            app=AppConfig(**config_data.get("app", {})),
            agent=AgentConfig(**config_data.get("agent", {})),
            memory=MemoryConfig(**config_data.get("memory", {})),
            tokens=TokensConfig(**config_data.get("tokens", {})),
            database=DatabaseConfig(**config_data.get("database", {})),
            chunking=ChunkingConfig(**config_data.get("chunking", {})),
            mindmap=MindmapConfig(**config_data.get("mindmap", {})),
            image=ImageConfig(**config_data.get("image", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )

        return settings

    def get_papers_folder_path(self) -> Path:
        """Get absolute path to papers folder."""
        path = Path(self.app.papers_folder)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    def get_vector_db_path(self) -> Path:
        """Get absolute path to vector database."""
        path = Path(self.database.vector_db_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    def get_conversation_db_path(self) -> Path:
        """Get absolute path to conversation database."""
        path = Path(self.database.conversation_db_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path


# Global settings instance
settings = Settings.load()
