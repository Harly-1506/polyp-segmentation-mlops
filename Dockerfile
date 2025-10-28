FROM python:3.12.9-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

FROM base AS deps

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group ml --no-dev


FROM deps AS runtime

COPY training/ ./training/
COPY .env ./


ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages"

RUN python -m ensurepip --upgrade

RUN python -c "import torch, ray, mlflow; \
print('torch', getattr(torch,'__version__','?'), \
'| ray', getattr(__import__('ray'),'__version__','?'), \
'| mlflow', getattr(__import__('mlflow'),'__version__','?'))"

