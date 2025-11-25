"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Optional
import tomllib

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
    host: str = "0.0.0.0"
    port: int = 8000


class AgentConfig(BaseModel):
    """AI Agent configuration."""

    model: str = "gemini-2.0-flash-exp"
    max_context_tokens: int = 1000000
    max_response_tokens: int = 30000
    temperature: float = 0.7
    # Optional separate model/config for orchestrator step
    orchestrator_model: Optional[str] = None
    orchestrator_temperature: float = 0.2
    orchestrator_max_output_tokens: int = 4096


class DatabaseConfig(BaseModel):
    """Database configuration."""

    vector_db_path: str = "./data/vectordb"
    conversation_db_path: str = "./data/conversations.db"


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_query: int = 5
    min_title_snippet_chars: int = 2000
    title_snippet_chars: int = 3000


class MindmapConfig(BaseModel):
    """Mindmap generation configuration."""

    max_depth: int = 4
    min_themes: int = 3
    max_themes: int = 7
    node_name_max_length: int = 60
    citation_node_min_size: int = 4
    citation_node_max_size: int = 15
    citation_count_max_threshold: int = 500


class ImageConfig(BaseModel):
    """Image generation configuration (Imagen)."""

    enabled: bool = False
    model: str = "imagegeneration"
    mime_type: str = "image/png"
    width: int = 1024
    height: int = 1024


class Settings(BaseModel):
    """Application settings loaded from config.toml and environment variables."""

    app: AppConfig = Field(default_factory=AppConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    mindmap: MindmapConfig = Field(default_factory=MindmapConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)

    # Environment variables
    google_api_key: str = Field(default="")

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from config.toml and environment variables."""
        if config_path is None:
            config_path = ROOT_DIR / "config.toml"

        # Load TOML configuration
        if config_path.exists():
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
        else:
            config_data = {}

        # Create settings instance
        settings = cls(
            app=AppConfig(**config_data.get("app", {})),
            agent=AgentConfig(**config_data.get("agent", {})),
            database=DatabaseConfig(**config_data.get("database", {})),
            chunking=ChunkingConfig(**config_data.get("chunking", {})),
            mindmap=MindmapConfig(**config_data.get("mindmap", {})),
            image=ImageConfig(**config_data.get("image", {})),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )

        return settings

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
