from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    dashscope_api_key: str = ""
    qwen_model: str = "qwen-plus"

    chroma_host: str = "localhost"
    chroma_port: int = 8001

    postgres_url: str = "postgresql://coach:coach@localhost:5432/adaptivecoach"


@lru_cache
def get_settings() -> Settings:
    return Settings()
