"""Configuration management for ESI StudyMate."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file if available
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseModel):
    """Application runtime configuration."""

    # LLM Provider settings
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower())
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    api_key: str = Field(default_factory=lambda: os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", "")))
    base_url: str = Field(default_factory=lambda: os.getenv("BASE_URL", ""))

    # Vector store & RAG settings
    chroma_persist_directory: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    )
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "600")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    top_k_results: int = Field(default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "3")))

    # Conversation memory
    max_short_term_messages: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SHORT_TERM_MESSAGES", "8"))
    )

    # Human-in-the-loop control
    require_retrieval_approval: bool = Field(
        default_factory=lambda: os.getenv("REQUIRE_RETRIEVAL_APPROVAL", "false").lower() in ("true", "1", "yes")
    )


settings = Settings()
