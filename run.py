"""CLI entry point to launch the JevK5 TypeSafe API Server."""

from __future__ import annotations

import argparse
import sys
import uvicorn

from jevk5_server.app import create_app
from jevk5_server.config import Settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JevK5 TypeSafe-compatible System One API Server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--llama-url",
        default="http://127.0.0.1:8080",
        help="llama-server HTTP base URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--model",
        default="jevk5-4b-v0.3-Q8_0",
        help="Model name or alias to advertise (default: jevk5-4b-v0.3-Q8_0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.22,
        help="Calibration temperature for options <= 16 (default: 1.22)",
    )
    parser.add_argument(
        "--knockout-temperature",
        type=float,
        default=0.93,
        help="Knockout tournament temperature for options > 16 (default: 0.93)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-K logprobs requested from llama-server (default: 40)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key required for Bearer authentication",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings(
        host=args.host,
        port=args.port,
        llama_server_url=args.llama_url,
        model_name=args.model,
        temperature=args.temperature,
        knockout_temperature=args.knockout_temperature,
        top_k=args.top_k,
        api_key=args.api_key,
    )
    app = create_app(settings=settings)

    print(
        f"Starting JevK5 TypeSafe API Server on http://{settings.host}:{settings.port}",
        flush=True,
    )
    print(f"Connecting to llama-server at {settings.llama_server_url}", flush=True)
    print(f"Serving model: {settings.model_name}", flush=True)

    uvicorn.run(app, host=settings.host, port=settings.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
