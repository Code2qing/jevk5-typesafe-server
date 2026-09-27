"""CLI entry point to launch the JevK5 TypeSafe API Server."""

from __future__ import annotations

import argparse
import logging
import sys
import uvicorn

from jevk5_server.app import create_app
from jevk5_server.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jevk5_server")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JevK5 TypeSafe-compatible System One API Server"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host interface to bind (default: $HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default: $PORT or 8000)",
    )
    parser.add_argument(
        "--llama-url",
        default=None,
        help="llama-server HTTP base URL (default: $LLAMA_SERVER_URL or http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name or alias to advertise (default: $MODEL_NAME or jevk5-4b-v0.3-Q8_0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Calibration temperature for options <= 16 (default: $TEMPERATURE or 1.22)",
    )
    parser.add_argument(
        "--knockout-temperature",
        type=float,
        default=None,
        help="Knockout tournament temperature for options > 16 (default: $KNOCKOUT_TEMPERATURE or 0.93)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K logprobs requested from llama-server (default: $TOP_K or 40)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key required for Bearer authentication (default: $API_KEY)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload",
    )
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Settings:
    """Build Settings respecting priority: CLI Arguments > Environment Variables > Defaults."""
    # Settings() automatically pulls from environment variables (os.getenv) with built-in defaults
    settings = Settings()

    if args.host is not None:
        settings.host = args.host
    if args.port is not None:
        settings.port = args.port
    if args.llama_url is not None:
        settings.llama_server_url = args.llama_url.rstrip("/")
    if args.model is not None:
        settings.model_name = args.model
    if args.temperature is not None:
        settings.temperature = args.temperature
    if args.knockout_temperature is not None:
        settings.knockout_temperature = args.knockout_temperature
    if args.top_k is not None:
        settings.top_k = args.top_k
    if args.api_key is not None:
        settings.api_key = args.api_key

    return settings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = build_settings(args)
    app = create_app(settings=settings)

    logger.info(
        "Starting JevK5 TypeSafe API Server on http://%s:%s",
        settings.host,
        settings.port,
    )
    logger.info("Connecting to llama-server at %s", settings.llama_server_url)
    logger.info("Serving model: %s", settings.model_name)

    uvicorn.run(app, host=settings.host, port=settings.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
