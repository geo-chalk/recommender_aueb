# --- pull official base image --- #
ARG BASEIMAGE=python:3.11-slim

FROM $BASEIMAGE as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV JUPYTER_PORT 8989

# ------ Set up workspace -------- #
RUN true && apt-get update
RUN apt update
RUN apt-get install -y curl
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y gcc


# Install system libs (LightGBM needs libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /workspace

# Copy dependency files first (Caching layer)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
# --frozen: Error if lockfile is out of sync
# --no-dev: Exclude dev dependencies (pytest, black, etc.)
# --no-install-project: Don't install the current project yet, just libs
# RUN uv sync --frozen --no-dev --no-install-project
RUN uv sync --frozen --no-install-project

# Enable the virtual environment
# uv creates a .venv folder. We just add it to the path.
ENV PATH="/workspace/.venv/bin:$PATH"

# Copy source code
COPY src/ ./src/

# Security: Non-root user
RUN useradd -m appuser && chown -R appuser /workspace
USER appuser


# --- Make temp file for source code --- #
COPY ./src /workspace/src
COPY ./data /workspace/data
COPY ./notebooks /workspace/notebooks

EXPOSE 8989

# --- Start Jupyter Notebook --- #
CMD ["uv", "run", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]