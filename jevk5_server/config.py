"""Server configuration settings."""

from __future__ import annotations

import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    llama_server_url: str = Field(
        default_factory=lambda: os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("MODEL_NAME", "jevk5-4b-v0.3-Q8_0")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "1.22"))
    )
    knockout_temperature: float = Field(
        default_factory=lambda: float(os.getenv("KNOCKOUT_TEMPERATURE", "0.93"))
    )
    top_k: int = Field(
        default_factory=lambda: int(os.getenv("TOP_K", "40"))
    )
    timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("TIMEOUT_S", "120.0"))
    )
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("API_KEY")
    )
    host: str = Field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("PORT", "8000"))
    )


settings = Settings()
