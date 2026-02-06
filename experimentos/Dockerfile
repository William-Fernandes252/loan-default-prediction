FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Build argument to enable/disable GPU support (default: 0 for CPU only)
ARG GPU=0

# - UV_COMPILE_BYTECODE: Compiles python files to .pyc for faster startup
# - UV_LINK_MODE: Copies files instead of hardlinking (safer for Docker layers)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    UV_NO_DEV=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gettext \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$GPU" = "1" ]; then \
    apt-get update && apt-get install -y --no-install-recommends wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       cuda-cudart-12-8 \
       libcublas-12-8 \
       libcudnn9-cuda-12 \
       libcusparse-12-8 \
       libcusolver-12-8 \
    && rm -rf /var/lib/apt/lists/*; \
    fi

COPY pyproject.toml uv.lock ./

# Install dependencies into the environment without installing the project itself yet.
# --no-install-project: Skips installing the root package, keeping this layer pure deps.
# If GPU=1, also install the gpu dependency group (cuML, GPU-enabled polars)
RUN if [ "$GPU" = "1" ]; then \
    uv sync --frozen --no-install-project --all-groups; \
    else \
    uv sync --frozen --no-install-project --all-groups --exclude gpu; \
    fi

COPY . .

RUN uv sync --frozen

RUN uv run python compile_i18n.py

RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "experiments.cli"]
CMD ["--help"]