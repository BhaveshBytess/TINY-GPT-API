"""
Centralized configuration loaded from environment variables.

Why a separate file?
- Single source of truth for all config
- Pydantic validates types at startup (not at request time)
- If a required key is missing, the server won't even start
  — better to fail fast than crash mid-request
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None  # NEW
    
    cloud_provider: str = "mistral"  # Changed default
    cloud_model: str = "mistral-small-latest"  # Or "mistral-medium-latest", "open-mistral-7b"

settings = Settings()