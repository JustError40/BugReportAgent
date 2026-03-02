FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (layer caching)
COPY pyproject.toml .
COPY langgraph.json .
COPY src/ src/

# Install dependencies
RUN uv pip install --system --no-cache-dir -e .

# Create state directory
RUN mkdir -p /app/state

# Non-root user
RUN useradd -m -u 1000 bugagent && chown -R bugagent:bugagent /app
USER bugagent

ENTRYPOINT ["python", "-m", "src.main"]
