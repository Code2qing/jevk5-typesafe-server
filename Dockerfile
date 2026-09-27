# Multi-stage lightweight Dockerfile using astral-sh/uv
FROM python:3.11-slim AS builder

# Install system dependencies needed for git-based dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency definitions and sync dependencies
COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen --no-install-project --no-dev

# Copy application code
COPY jevk5_server /app/jevk5_server
COPY run.py /app/run.py
RUN uv sync --frozen --no-dev

# Final production image
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and app code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/jevk5_server /app/jevk5_server
COPY --from=builder /app/run.py /app/run.py

# Put virtualenv into PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run.py"]
