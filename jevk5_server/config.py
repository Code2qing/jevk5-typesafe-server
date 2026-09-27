"""Server configuration settings."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llama_server_url: str = Field(
        default="http://127.0.0.1:8080",
        description="Base URL of the running llama-server instance.",
    )
    model_name: str = Field(
        default="jevk5-4b-v0.3-Q8_0",
        description="Advertised model identifier.",
    )
    temperature: float = Field(
        default=1.22,
        description="Temperature for options <= 16.",
    )
    knockout_temperature: float = Field(
        default=0.93,
        description="Knockout tournament temperature for options > 16.",
    )
    top_k: int = Field(
        default=40,
        description="Number of top logprobs to request from llama-server.",
    )
    timeout_s: float = Field(
        default=120.0,
        description="HTTP request timeout in seconds.",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key for Bearer authentication.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Server bind host.",
    )
    port: int = Field(
        default=8000,
        description="Server bind port.",
    )

    @field_validator("llama_server_url")
    @classmethod
    def clean_url(cls, v: str) -> str:
        return v.rstrip("/")


settings = Settings()
