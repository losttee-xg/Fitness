from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RALLY_", env_file=".env", extra="ignore")

    app_name: str = "Rally Health API"
    environment: str = "development"
    api_key: str = ""
    database_path: Path = Path("data/rally_health.sqlite3")
    normalized_data_dir: Path = Path("data/normalized")
    source_dir: Path = Path(".")
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_enabled: bool = False
    chat_enabled: bool = False
    chat_api_key: str | None = None
    chat_base_url: str = "https://api.pinkyne.com/v1"
    chat_model: str = "gpt-4o"
    chat_timeout_seconds: int = Field(default=90, ge=5, le=300)
    chat_history_turns: int = Field(default=12, ge=0, le=40)
    ocr_enabled: bool = False
    ocr_api_key: str | None = None
    ocr_base_url: str = "https://api.pinkyne.com/v1"
    ocr_model: str = "gpt-4o"
    ocr_timeout_seconds: int = Field(default=90, ge=5, le=300)
    ocr_test_allow_billable: bool = False
    max_chat_input_chars: int = Field(default=4000, ge=100, le=20000)
    allowed_origins: list[str] = []

    @property
    def is_production(self) -> bool:
        return self.environment.casefold() == "production"

    def validate_runtime(self) -> None:
        if self.is_production and not self.api_key:
            raise RuntimeError("RALLY_API_KEY must be configured in production")
        if self.anthropic_enabled and not self.anthropic_api_key:
            raise RuntimeError("RALLY_ANTHROPIC_API_KEY is required when AI is enabled")
        if self.chat_enabled and not self.chat_api_key:
            raise RuntimeError("RALLY_CHAT_API_KEY is required when GPT-4o chat is enabled")
        if self.ocr_enabled and not self.ocr_api_key:
            raise RuntimeError("RALLY_OCR_API_KEY is required when report OCR is enabled")
        if self.ocr_enabled and self.ocr_model != "gpt-4o":
            raise RuntimeError("Report OCR is pinned to gpt-4o")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_runtime()
    return settings
