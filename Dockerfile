# Build stage: uv and a system Python, discarded once the venv is built.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Precompile bytecode: read-only Singularity can't cache it at runtime.
ENV UV_COMPILE_BYTECODE=1

# Copy out of the cache mount, since hardlinks don't cross filesystems.
ENV UV_LINK_MODE=copy

# Use the system Python, which the runtime stage also has.
ENV UV_PYTHON_DOWNLOADS=never

# Assemble here so only the venv needs copying later.
WORKDIR /app

# Dependencies in their own layer so source edits don't refetch torch.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project --no-dev

# Hatchling reads README and LICENSE for wheel metadata.
COPY pyproject.toml uv.lock README.md LICENSE ./

# Copied last to keep the dependency layer cached.
COPY src ./src

# Non-editable, so the venv doesn't depend on /app/src.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Runtime stage
FROM python:3.13-slim-bookworm

# ps, for trace metrics.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

# The venv is all that's worth carrying over.
COPY --from=builder /app/.venv /app/.venv

# On PATH, not activated: singularity exec ignores ENTRYPOINT.
ENV PATH=/app/.venv/bin:$PATH

# Stops the host's ~/.local shadowing ours when Apptainer mounts $HOME.
ENV PYTHONNOUSERSITE=1

# Default directory for bind-mounted inputs and outputs.
WORKDIR /work


CMD ["pydiffuse", "--help"]
