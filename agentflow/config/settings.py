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
