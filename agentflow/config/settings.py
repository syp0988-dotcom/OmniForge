from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    model_config = ConfigDict(env_file=None, extra="ignore")

    app_name: str = Field(default="OmniForge", alias="APP_NAME")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    model_name: str = Field(default="deepseek-chat", alias="MODEL_NAME")
    temperature: float = Field(default=0.2, alias="TEMPERATURE")
    max_tokens: int = Field(default=1000, alias="MAX_TOKENS")

    # -- Knowledge base settings --
    knowledge_embedder: str = Field(default="tfidf", alias="KNOWLEDGE_EMBEDDER")
    knowledge_embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        alias="KNOWLEDGE_EMBEDDING_MODEL",
    )
    knowledge_alpha: float = Field(default=0.7, alias="KNOWLEDGE_ALPHA")
    knowledge_beta: float = Field(default=0.3, alias="KNOWLEDGE_BETA")
    knowledge_chunk_size: int = Field(default=500, alias="KNOWLEDGE_CHUNK_SIZE")
    knowledge_chunk_overlap: int = Field(default=50, alias="KNOWLEDGE_CHUNK_OVERLAP")
    knowledge_chroma_path: str = Field(default="", alias="KNOWLEDGE_CHROMA_PATH")
    knowledge_chroma_collection: str = Field(default="knowledge_chunks", alias="KNOWLEDGE_CHROMA_COLLECTION")
    knowledge_top_k: int = Field(default=5, alias="KNOWLEDGE_TOP_K")
    knowledge_min_score: float = Field(default=0.05, alias="KNOWLEDGE_MIN_SCORE")

    # -- Context window & truncation settings --
    max_context_chars: int = Field(default=12000, alias="MAX_CONTEXT_CHARS")
    max_workspace_files: int = Field(default=50, alias="MAX_WORKSPACE_FILES")
    max_search_results: int = Field(default=5, alias="MAX_SEARCH_RESULTS")

    # -- Session timeout & cleanup settings --
    session_ttl_hours: int = Field(default=72, alias="SESSION_TTL_HOURS")
    cleanup_interval_minutes: int = Field(default=60, alias="CLEANUP_INTERVAL_MINUTES")
    memory_ttl_days: int = Field(default=30, alias="MEMORY_TTL_DAYS")

    # -- Token budget settings --
    max_session_tokens: int = Field(default=50000, alias="MAX_SESSION_TOKENS")

    # -- Tool safety settings --
    allow_unsafe_python_tool: bool = Field(default=False, alias="ALLOW_UNSAFE_PYTHON_TOOL")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def database_path(self) -> Path:
        # Keep database path stable to avoid migration issues during rename.
        return self.project_root / "agentflow" / "database" / "agentflow.db"


settings = Settings()
