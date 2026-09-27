"""FastAPI application implementing the TypeSafe AI System One specification."""

from __future__ import annotations

import urllib.request
import urllib.error
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from jevk5 import JevK5GGUF

from .adapter import JevK5TypeSafeAdapter
from .config import Settings, settings as default_settings
from .schemas import (
    ModelCard,
    ModelsResponse,
    SystemOneRequest,
    SystemOneResponse,
)


def create_app(
    settings: Settings | None = None,
    adapter: JevK5TypeSafeAdapter | None = None,
) -> FastAPI:
    cfg = settings or default_settings

    if adapter is None:
        model = JevK5GGUF(
            url=cfg.llama_server_url,
            temperature=cfg.temperature,
            knockout_temperature=cfg.knockout_temperature,
            top_k=cfg.top_k,
            timeout_s=cfg.timeout_s,
        )
        adapter = JevK5TypeSafeAdapter(model=model, default_model_name=cfg.model_name)

    app = FastAPI(
        title="JevK5 TypeSafe API Server",
        description="TypeSafe-compatible evaluation server powered by JevK5 on llama-server",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def verify_auth(authorization: str | None) -> None:
        if not cfg.api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "Missing or invalid API key. Check the Authorization header."},
            )
        token = authorization.split("Bearer ", 1)[1].strip()
        if token != cfg.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "Invalid API key."},
            )

    @app.get("/health")
    def health_check():
        llama_healthy = adapter.check_health()
        return {
            "ok": True,
            "status": "healthy" if llama_healthy else "degraded",
            "backend_llama_connected": llama_healthy,
            "model": cfg.model_name,
        }

    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models(authorization: str | None = Header(None)):
        verify_auth(authorization)
        return ModelsResponse(
            models=[
                ModelCard(
                    name="jev-latest",
                    description="The most recent stable release of JevK5.",
                    release_date="2026-09-01",
                ),
                ModelCard(
                    name="jev-preview",
                    description="Preview build for JevK5 decisions.",
                    release_date="2026-09-24",
                ),
                ModelCard(
                    name=cfg.model_name,
                    description="Active local GGUF model running on llama-server.",
                    release_date="2026-09-24",
                ),
            ]
        )

    @app.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        responses={
            401: {"description": "Missing or invalid API key"},
            422: {"description": "Validation error in request body or criteria"},
            502: {"description": "Backend llama-server connection failure"},
        },
    )
    async def system_one(
        request: SystemOneRequest,
        authorization: str | None = Header(None),
    ):
        verify_auth(authorization)
        try:
            return await adapter.evaluate(request)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error": f"Failed to connect to backend llama-server: {exc}"},
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Evaluation error: {exc}"},
            )

    return app


app = create_app()
