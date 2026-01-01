# Use the official uv image as a parent image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 1. Setup Environment
# - UV_COMPILE_BYTECODE: Compiles python files to .pyc for faster startup
# - UV_LINK_MODE: Copies files instead of hardlinking (safer for Docker layers)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    UV_NO_DEV=1

WORKDIR /app

# 2. Install System Dependencies
# Only install what is strictly needed for runtime or wheel building.
# We clean up apt lists to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gettext \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python Dependencies (Cached Layer)
# We copy ONLY the dependency files first. This layer is cached until
# uv.lock or pyproject.toml changes.
COPY pyproject.toml uv.lock ./

# Install dependencies into the environment without installing the project itself yet.
# --no-install-project: Skips installing the root package, keeping this layer pure deps.
RUN uv sync --frozen --no-install-project

# 4. Install Application Code
# This layer changes frequently, but because it's after the heavy install,
# rebuilds are fast.
COPY . .

# Now install the project itself (creates the package metadata).
RUN uv sync --frozen

# 5. Build Artifacts
# Compile translation files (.po -> .mo)
RUN uv run python compile_i18n.py

# 6. Security: Create a non-root user
# Create a user 'appuser' and switch to it.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# 7. Runtime Configuration
# Add the virtual environment to the PATH so we can use 'python' directly.
ENV PATH="/app/.venv/bin:$PATH"

# Run the module directly.
ENTRYPOINT ["python", "-m", "experiments.cli"]
CMD ["--help"]