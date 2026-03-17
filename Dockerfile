# ============================================================
# Book Detection API — Production Dockerfile
# ============================================================
# Build:   docker build -t book-detection-api .
# Run:     docker run -p 8000:8000 --env-file .env book-detection-api
# ============================================================

FROM python:3.11-slim

# System libraries required by Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy requirements first for Docker layer caching ──────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create data directory for ChromaDB persistence
RUN mkdir -p data/chroma_db

# Expose API port
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ─────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
