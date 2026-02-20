# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system deps (combine apt into one layer)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (best cache behaviour)
COPY requirements.txt .

# Use BuildKit pip cache (DO NOT use --no-cache-dir / PIP_NO_CACHE_DIR)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy app code after deps
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
