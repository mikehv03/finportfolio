FROM python:3.11-slim

LABEL maintainer="Miguel Herrera"
LABEL description="finportfolio — Portfolio Theory and Asset Pricing Library"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./

COPY finportfolio/ finportfolio/
COPY tests/ tests/
COPY notebooks/ notebooks/
COPY scripts/ scripts/

RUN pip install --upgrade pip && \
    pip install ".[dev]"

CMD ["python", "scripts/example.py"]
