# ============================================================
# Book Detection API — Production Dockerfile
# ============================================================
# Build:   docker build -t book-detection-api .
# Run:     docker run -p 8000:8000 --env-file .env book-detection-api
# ============================================================

FROM python:3.11-slim

# System libraries required by Pillow and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy requirements first for Docker layer caching ──────────
# Changes to requirements.txt invalidate this layer.
# Changes to app/ code do NOT invalidate it — fast rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model into the image ───────────
# This bakes the 80MB sentence-transformer model into the image layer.
# Avoids runtime downloads from HuggingFace on every container start.
# If you change EMBEDDING_MODEL, update this line to match.
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2'); \
               print('Embedding model cached.')"

# ── Copy application source ───────────────────────────────────
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create data directory for ChromaDB persistence
# (the actual data comes from a mounted Docker volume — see docker-compose.yml)
RUN mkdir -p data/chroma_db

# Expose API port
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────
# Docker monitors this — container is "healthy" once the API responds.
# Start period gives time for the sentence-transformer model to load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ─────────────────────────────────────────────
# Single worker — Langfuse v3 uses OpenTelemetry which is not fork-safe.
# Multiple workers via os.fork() inherit stale OTel state and crash.
# asyncio handles I/O concurrency (OpenAI API calls) just fine.
# For horizontal scaling, run multiple container replicas instead.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
