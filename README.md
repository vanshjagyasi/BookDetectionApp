# Book Detection API

A production-ready REST API that identifies books from cover photos using a multimodal RAG pipeline. Upload any book cover image (front, back, or side) and receive structured JSON metadata.

**Built with:** FastAPI · OpenAI GPT-4o · ChromaDB · sentence-transformers · Pydantic v2 · React · Docker

---

## Architecture Overview

```
┌─────────────┐     POST /api/v1/detect-book      ┌──────────────────────────────────────┐
│             │ ─────── image file ──────────────▶ │           FastAPI                    │
│   Client    │                                    │                                      │
│             │ ◀──────── BookInfo JSON ─────────  │  ┌───────────┐                       │
└─────────────┘                                    │  │  Vision   │ GPT-4o reads image    │
                                                   │  │  Service  │ → VisionExtraction    │
                                                   │  └─────┬─────┘                       │
                                                   │        │                              │
                                                   │  ┌─────▼─────┐                       │
                                                   │  │   RAG     │ ChromaDB → 10 cands   │
                                                   │  │  Service  │ re-rank → top 3       │
                                                   │  └─────┬─────┘                       │
                                                   │        │                              │
                                                   │  ┌─────▼─────┐                       │
                                                   │  │   LLM     │ GPT-4o synthesises    │
                                                   │  │  Service  │ → BookInfo JSON       │
                                                   │  └───────────┘                       │
                                                   └──────────────────────────────────────┘
```

### Pipeline Stages

| Stage | File | What it does |
|-------|------|--------------|
| 1. Vision | `app/services/vision_service.py` | GPT-4o multimodal locates the book in the image, extracts title, author, ISBN, text |
| 2. RAG | `app/services/rag_service.py` | Embeds extracted text, queries ChromaDB for 10 candidates, cross-encoder re-ranks to top 3 |
| 3. LLM | `app/services/llm_service.py` | GPT-4o cross-references extraction + RAG candidates → structured BookInfo |

---

## Prerequisites

- Python 3.11+
- An **OpenAI API key** (requires access to `gpt-4o`)
- A **Google Books API key** (only needed to seed the database once)
- Docker + Docker Compose (optional, for containerised deployment)

---

## Quick Start

```bash
# 1. Clone and enter directory
git clone <repo-url>
cd BookDetectionNew

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and set:
#   OPENAI_API_KEY=sk-...
#   GOOGLE_BOOKS_API_KEY=AIzaSy-...

# 5. Seed the book database (run once)
python scripts/populate_db.py
# Output: ~100 books stored in ChromaDB

# 6. Start the API
uvicorn app.main:app --reload
# API running at http://localhost:8000
# Swagger UI at http://localhost:8000/docs

# 7. Test with a book cover image
curl -X POST http://localhost:8000/api/v1/detect-book \
     -F "file=@path/to/book_cover.jpg"
```

---

## Configuration

All settings are read from the `.env` file (copy from `.env.example`).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key. Get one at platform.openai.com |
| `OPENAI_MODEL` | No | `gpt-4o` | Model for vision + generation. `gpt-4o-mini` is cheaper but less accurate |
| `GOOGLE_BOOKS_API_KEY` | For seeding | — | Required only for `scripts/populate_db.py` |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma_db` | Where ChromaDB stores data on disk |
| `CHROMA_COLLECTION_NAME` | No | `books` | ChromaDB collection name |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Bi-encoder model. Must stay consistent with what seeded the DB |
| `RAG_FETCH_K` | No | `10` | Candidates to fetch from ChromaDB before cross-encoder re-ranking |
| `RERANKER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for re-ranking. Set to `""` to disable |
| `RERANK_TOP_K` | No | `3` | Final number of re-ranked candidates passed to GPT-4o |
| `LANGFUSE_PUBLIC_KEY` | No | `""` | Langfuse project public key (traces disabled if empty) |
| `LANGFUSE_SECRET_KEY` | No | `""` | Langfuse project secret key |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse server URL (change for self-hosted) |
| `MAX_IMAGE_SIZE_MB` | No | `20` | Maximum allowed image upload size |
| `ALLOWED_ORIGINS` | No | `["*"]` | CORS allowed origins. Restrict in production |

---

## API Reference

### `POST /api/v1/detect-book`

Identify a book from its cover image.

**Request**
```
Content-Type: multipart/form-data
Body: file=<image file>
```

Accepted image formats: JPEG, PNG, WEBP, GIF
Maximum size: `MAX_IMAGE_SIZE_MB` (default 20MB)

**Response `200 OK`**
```json
{
  "success": true,
  "book": {
    "title": "Dune",
    "author": "Frank Herbert",
    "isbn": "9780441013593",
    "publisher": "Ace",
    "publication_year": 1965,
    "genre": "Science Fiction",
    "tags": ["space opera", "dystopian", "political thriller"],
    "synopsis": "A desert planet, a noble family betrayed, and the most important substance in the universe.",
    "price": 9.99,
    "rating": 4.5,
    "confidence_score": 0.92
  },
  "extraction_notes": "Vision read: 'DUNE' | Author: 'Frank Herbert' | RAG matched 3 candidate(s)"
}
```

**Confidence Score Guide**

| Score | Meaning |
|-------|---------|
| ≥ 0.95 | ISBN visible in image AND matches a database record exactly |
| 0.90–0.94 | Title AND author clearly readable, database candidate matches with good similarity (≥ 0.60) |
| 0.80–0.89 | Title AND author clearly readable, database match with lower similarity |
| 0.70–0.79 | Title AND author clearly readable, even without a database match |
| 0.40–0.69 | Only title OR author readable (partial signal) |
| < 0.40 | No text evidence; identification from visual inference alone |

**Error Responses**

| Status | Code | When |
|--------|------|------|
| 413 | — | Image exceeds `MAX_IMAGE_SIZE_MB` |
| 415 | — | Unsupported image format |
| 422 | — | Corrupt or unreadable image file |
| 500 | `INTERNAL_SERVER_ERROR` | OpenAI API failure or unexpected error |

### `GET /health`

Health check — returns API status and book count.

```json
{"status": "ok", "books_in_db": 97, "model": "gpt-4o"}
```

---

## Project Structure

```
BookDetectionNew/
├── app/
│   ├── main.py              Entry point. FastAPI factory + lifespan (loads VectorStore once)
│   ├── config.py            Pydantic Settings — all env vars, @lru_cache singleton
│   ├── dependencies.py      FastAPI DI helpers (get_vector_store, get_reranker)
│   │
│   ├── api/v1/routes/
│   │   └── detection.py     POST /api/v1/detect-book — orchestrates all 3 stages
│   │
│   ├── schemas/
│   │   ├── book.py          VisionExtraction, BookInfo, DetectionResponse models
│   │   └── errors.py        ErrorDetail, ErrorResponse models
│   │
│   ├── services/
│   │   ├── vision_service.py   Stage 1: GPT-4o reads image → VisionExtraction
│   │   ├── rag_service.py      Stage 2: bi-encoder search + cross-encoder re-rank
│   │   └── llm_service.py      Stage 3: GPT-4o synthesises → BookInfo JSON
│   │
│   ├── db/
│   │   └── vector_store.py  ChromaDB PersistentClient + SentenceTransformer
│   │
│   └── utils/
│       ├── image_utils.py   Validate, resize (≤2048px), base64-encode images
│       └── text_utils.py    clean_text(), truncate() helpers
│
├── frontend/                React SPA (mobile-friendly book scanner UI)
│   ├── src/
│   │   ├── App.tsx          Root component: camera → loading → result/error
│   │   ├── types.ts         BookInfo, DetectionResponse interfaces
│   │   ├── api/
│   │   │   └── detectBook.ts   POST /api/v1/detect-book via fetch
│   │   ├── hooks/
│   │   │   └── useBookDetection.ts   State machine (idle/loading/success/error)
│   │   └── components/
│   │       ├── CameraCapture.tsx     Camera input + file picker
│   │       ├── LoadingSpinner.tsx    Scan animation over preview
│   │       ├── BookResultCard.tsx    Full metadata display
│   │       ├── ConfidenceBadge.tsx   Color-coded confidence indicator
│   │       └── ErrorMessage.tsx      Error card + retry
│   ├── Dockerfile           Multi-stage: node build → nginx serve
│   ├── nginx.conf           SPA serving + /api/ proxy to backend
│   └── package.json         React 18 + Vite + Tailwind CSS
│
├── scripts/
│   ├── populate_db.py       Google Books API → ChromaDB ingestion (~100 books)
│   └── verify_db.py         Inspect ChromaDB: list books, run semantic search
│
├── tests/
│   ├── conftest.py          Pytest fixtures + mock OpenAI/ChromaDB clients
│   ├── test_detection.py    Integration test for /detect-book endpoint
│   ├── test_vision_service.py  Unit tests for VisionService
│   └── test_rag_service.py     Unit tests for RAGService
│
├── docs/
│   ├── PIPELINE.md          Deep-dive: pipeline stages, data flow, design decisions
│   ├── EXTENDING.md         How to swap AI providers, add fields, add endpoints
│   └── API_REFERENCE.md     Full API docs with curl examples for every response type
│
├── data/chroma_db/          ChromaDB persistent storage (gitignored)
├── CLAUDE.md                AI model navigation guide (read this first when using an AI assistant)
├── .env.example             Template environment file
├── requirements.txt         Python dependencies with pinned versions
├── Dockerfile               Production Docker image (API)
└── docker-compose.yml       API + frontend + optional DB seeder
```

---

## Database Seeding

The book database is populated by `scripts/populate_db.py` using the Google Books API.

**What it fetches:** ~100 books across 12 genre queries (fiction, sci-fi, mystery, biography, programming, ML/AI, fantasy, romance, history, self-help, young adult).

**Re-running:** Safe to re-run — all inserts are upserts (idempotent by ISBN or title hash).

```bash
# Seed (requires GOOGLE_BOOKS_API_KEY in .env)
python scripts/populate_db.py

# Verify contents
python scripts/verify_db.py              # list first 10 books
python scripts/verify_db.py --all        # list all books
python scripts/verify_db.py --query "harry potter wizard"  # semantic search test
```

---

## Running Tests

```bash
pytest tests/ -v                    # run all tests
pytest tests/test_detection.py -v   # run endpoint tests only
pytest tests/ -v --tb=short         # short traceback on failure
```

Tests use mocked OpenAI clients and an in-memory ChromaDB instance —
no real API calls or disk I/O are required to run the test suite.

---

## Docker Deployment

```bash
# Build all images (API + frontend)
docker-compose build

# Seed the database (run once; data persists in 'chroma_data' Docker volume)
docker-compose --profile seed up db-seeder

# Start everything (API + frontend)
docker-compose up

# Or start individually
docker-compose up api       # API only at http://localhost:8000
docker-compose up frontend  # Frontend only (requires healthy API)
```

After startup:
- **Frontend** — `http://localhost:3000` (React SPA served by nginx)
- **API** — `http://localhost:8000` (FastAPI, Swagger at `/docs`)
- **On mobile** — open `http://<your-local-ip>:3000` on the same WiFi

The frontend's nginx reverse-proxies `/api/` requests to the API container, so everything is same-origin (no CORS issues). ChromaDB data is persisted in a named Docker volume (`chroma_data`) and survives container restarts.

**Note:** The API Dockerfile runs a single uvicorn worker (no `--workers`). Langfuse v3 uses OpenTelemetry which is not fork-safe — multiple workers via `os.fork()` cause crashes. For horizontal scaling, run multiple container replicas instead.

---

## Frontend

A mobile-friendly React SPA lives in `frontend/`. It opens the phone's rear camera, captures a book cover photo, sends it to the API, and displays the identified book metadata.

**Stack:** React 18 · TypeScript · Vite · Tailwind CSS · nginx

```bash
# Local development (requires API running at localhost:8000)
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173 with Vite dev proxy to API

# Production (via Docker)
docker-compose up   # frontend at :3000, API at :8000
```

**Key features:**
- `capture="environment"` opens the rear camera on mobile, falls back to file picker on desktop
- 120s request timeout (GPT-4o pipeline takes 10–30s)
- Client-side file validation (<20MB, image MIME types only)
- Scanning animation while identifying
- Full metadata display with confidence badge, star ratings, tags, collapsible synopsis

---

## Swapping AI Providers

The vision and LLM services are intentionally isolated. To swap from OpenAI to another provider, see [docs/EXTENDING.md](docs/EXTENDING.md).

Quick summary of what changes:
- **Anthropic Claude**: `vision_service.py` (different image format), `llm_service.py` (tool_use instead of beta.parse), `config.py`, `requirements.txt`
- **Google Gemini**: Similar to Claude changes
- **Local/Ollama**: Change `base_url` in the OpenAI client — the API is compatible

---

## Extending the Project

See [docs/EXTENDING.md](docs/EXTENDING.md) for step-by-step guides on:
- Adding new fields to the BookInfo response
- Changing the embedding model
- Adding new API endpoints
- Scaling with a managed vector database (Pinecone, Qdrant)
