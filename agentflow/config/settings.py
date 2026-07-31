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
    log_format: str = Field(default="text", alias="LOG_FORMAT")  # "text" | "json"
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    model_name: str = Field(default="deepseek-chat", alias="MODEL_NAME")
    temperature: float = Field(default=0.2, alias="TEMPERATURE")
    max_tokens: int = Field(default=1000, alias="MAX_TOKENS")

    # -- Knowledge base settings --
    knowledge_alpha: float = Field(default=0.7, alias="KNOWLEDGE_ALPHA")
    knowledge_beta: float = Field(default=0.3, alias="KNOWLEDGE_BETA")
    knowledge_chunk_size: int = Field(default=500, alias="KNOWLEDGE_CHUNK_SIZE")
    knowledge_chunk_overlap: int = Field(default=50, alias="KNOWLEDGE_CHUNK_OVERLAP")
    knowledge_top_k: int = Field(default=5, alias="KNOWLEDGE_TOP_K")
    # Hybrid-score floor. 0.05 admitted pure-lexical noise (vector_score=0,
    # score ~0.06); 0.10 keeps semantically relevant chunks while filtering
    # keyword-only matches. Tune via agentflow.knowledge.eval.tune.
    knowledge_min_score: float = Field(default=0.10, alias="KNOWLEDGE_MIN_SCORE")
    # Post-retrieval reranking: "none" (default) or "llm".
    # When "llm", top candidates are re-ranked by the LLM before answering.
    knowledge_reranker: str = Field(default="none", alias="KNOWLEDGE_RERANKER")
    knowledge_rerank_candidates: int = Field(
        default=10, alias="KNOWLEDGE_RERANK_CANDIDATES",
    )
    knowledge_rerank_top_k: int = Field(
        default=3, alias="KNOWLEDGE_RERANK_TOP_K",
    )

    # -- Embedding API settings (DashScope / OpenAI-compatible) --
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="EMBEDDING_BASE_URL",
    )
    embedding_model_name: str = Field(
        default="text-embedding-v3", alias="EMBEDDING_MODEL_NAME"
    )
    embedding_cache_enabled: bool = Field(default=True, alias="EMBEDDING_CACHE_ENABLED")
    embedding_cache_path: str = Field(default="", alias="EMBEDDING_CACHE_PATH")

    # -- Qdrant settings --
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="knowledge_chunks", alias="QDRANT_COLLECTION")

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

    # -- Request / upload limits --
    # Safety net for the whole chat request (streaming included). Default 300s
    # is generous for multi-step workflows; single LLM calls have their own 60s
    # client timeout plus retries.
    max_request_seconds: int = Field(default=300, alias="MAX_REQUEST_SECONDS")
    max_upload_bytes: int = Field(
        default=50 * 1024 * 1024, alias="MAX_UPLOAD_BYTES",
    )
    max_zip_entries: int = Field(default=200, alias="MAX_ZIP_ENTRIES")
    max_zip_uncompressed_bytes: int = Field(
        default=500 * 1024 * 1024, alias="MAX_ZIP_UNCOMPRESSED_BYTES",
    )

    # -- Deployment security --
    # When AUTH_TOKEN is set, every endpoint except /health requires
    # "Authorization: Bearer <token>". Leave empty for local-only use.
    auth_token: str = Field(default="", alias="AUTH_TOKEN")
    # Comma-separated list of allowed CORS origins, e.g.
    # "https://app.example.com,https://admin.example.com".
    # Empty → default localhost-only regex (safe for local dev).
    cors_origins: str = Field(default="", alias="CORS_ORIGINS")

    # -- HTTP proxy settings (for search providers behind GFW) --
    http_proxy: str = Field(default="", alias="HTTP_PROXY")
    https_proxy: str = Field(default="", alias="HTTPS_PROXY")

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
