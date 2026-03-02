# CLAUDE.md — AI Model Navigation Guide

## What is this project?
A FastAPI REST service that accepts a book cover photo (front, back, or side), runs it through GPT-4o vision to extract visible text, queries a ChromaDB vector store for matching books, then uses GPT-4o Structured Outputs to return a fully validated JSON object with book metadata.

**Stack:** Python 3.11 · FastAPI · OpenAI GPT-4o · ChromaDB · sentence-transformers · Pydantic v2 · React · Docker

---

## Critical File Map

| Task | File to open |
|------|-------------|
| Change JSON output fields | [app/schemas/book.py](app/schemas/book.py) → `BookInfo` class |
| Change vision extraction prompt | [app/services/vision_service.py](app/services/vision_service.py) → `VISION_SYSTEM_PROMPT` |
| Change LLM synthesis prompt / confidence rules | [app/services/llm_service.py](app/services/llm_service.py) → `LLM_SYSTEM_PROMPT` |
| Change RAG retrieval count | `.env` → `RAG_FETCH_K` (bi-encoder) and `RERANK_TOP_K` (final to LLM) |
| Change re-ranker model / disable re-ranking | `.env` → `RERANKER_MODEL` (set `""` to disable) |
| Change vector similarity logic | [app/db/vector_store.py](app/db/vector_store.py) → `query()` |
| Configure Langfuse tracing | `.env` → `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |
| Modify frontend UI | [frontend/src/](frontend/src/) — React 18 + Vite + Tailwind CSS |
| Add a new API endpoint | Create file in [app/api/v1/routes/](app/api/v1/routes/) and register in [app/main.py](app/main.py) |
| Add or change config variables | [app/config.py](app/config.py) → `Settings` class + [.env.example](.env.example) |
| Swap AI provider (OpenAI → Claude / Gemini / Ollama) | [docs/EXTENDING.md](docs/EXTENDING.md) |
| Seed / rebuild the book database | `python scripts/populate_db.py` |
| Inspect what is in ChromaDB | `python scripts/verify_db.py` |
| Change the embedding model | [app/config.py](app/config.py) `EMBEDDING_MODEL` **then** re-run populate_db.py |
| Understand the full pipeline | [docs/PIPELINE.md](docs/PIPELINE.md) |
| API request/response format | [docs/API_REFERENCE.md](docs/API_REFERENCE.md) |

---

## Run Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env — add OPENAI_API_KEY and GOOGLE_BOOKS_API_KEY

# 3. Seed the database (run once)
python scripts/populate_db.py

# 4. Start the API
uvicorn app.main:app --reload

# 5. Test it
curl -X POST http://localhost:8000/api/v1/detect-book \
     -F "file=@path/to/book_cover.jpg"

# Open Swagger UI
# http://localhost:8000/docs

# Frontend (local dev — requires API at localhost:8000)
cd frontend && npm install && npm run dev

# Docker (full stack — API + frontend)
docker-compose build
docker-compose --profile seed up db-seeder   # seed once
docker-compose up                            # API at :8000, frontend at :3000

# Tests
pytest tests/ -v
```

---

## Architecture Invariants

These are things that must NOT be changed without understanding the consequences:

| Invariant | Reason |
|-----------|--------|
| `VectorStore` is created ONCE in `main.py` `lifespan()` | The sentence-transformer model is ~80MB — loading it per-request would cause multi-second latency on every call |
| `CrossEncoder` is created ONCE in `main.py` `lifespan()` | Same reason as VectorStore — ~80MB model, loaded once at startup |
| ChromaDB collection uses `hnsw:space=cosine` | sentence-transformers produces L2-normalised vectors; cosine is the correct metric. Changing this after population corrupts all similarity scores |
| `EMBEDDING_MODEL` must stay consistent with what populated the DB | Embeddings from different models are incompatible — changing the model requires re-running `populate_db.py` |
| `llm_service.py` uses `beta.chat.completions.parse(response_format=BookInfo)` | This is OpenAI Structured Outputs — it guarantees schema-valid output. Do NOT switch to plain text completion |
| Single uvicorn worker (no `--workers`) | Langfuse v3 uses OpenTelemetry which is not fork-safe. Multiple workers via `os.fork()` crash. Scale with container replicas instead |
| Service calls wrapped in `asyncio.to_thread()` | `langfuse.openai.OpenAI` wrapper uses asyncio internally — calling it synchronously from uvicorn's event loop causes deadlocks |

---

## Pipeline Summary (3 stages)

```
Image upload
    │
    ▼ Stage 1 — vision_service.py
    GPT-4o locates the book in the image, ignores background
    → VisionExtraction {visible_title, visible_author, visible_isbn, other_text, cover_description}
    │
    ▼ Stage 2 — rag_service.py
    Build query string → embed with all-MiniLM-L6-v2
    → ChromaDB cosine search → 10 candidates
    → CrossEncoder re-rank → top 3 candidates
    │
    ▼ Stage 3 — llm_service.py
    GPT-4o sees: VisionExtraction + RAG candidates
    → Structured Outputs → BookInfo (title, author, isbn, … confidence_score)
    │
    ▼ HTTP 200 DetectionResponse JSON
```

---

## Common Tasks (Step-by-Step)

### Add a new field to the JSON output
1. Add field to `BookInfo` in [app/schemas/book.py](app/schemas/book.py)
2. Update `LLM_SYSTEM_PROMPT` in [app/services/llm_service.py](app/services/llm_service.py) to instruct the model to populate it
3. If sourced from the DB, add it to `metadata` dict in [scripts/populate_db.py](scripts/populate_db.py) and re-run populate

### Swap OpenAI for Anthropic Claude
See [docs/EXTENDING.md](docs/EXTENDING.md) → "Switching to Anthropic Claude"

### Change the number of RAG candidates
- `RAG_FETCH_K=10` — candidates fetched from ChromaDB (bi-encoder). Increase for better recall.
- `RERANK_TOP_K=3` — final candidates after cross-encoder re-ranking (sent to GPT-4o). More = higher token cost.
- `RERANKER_MODEL=""` — set empty to disable re-ranking entirely (uses bi-encoder order).

### Re-seed the database after adding new genres
Edit `SEARCH_QUERIES` list in [scripts/populate_db.py](scripts/populate_db.py), then run:
```bash
python scripts/populate_db.py
```
It is safe to re-run — existing entries are upserted (not duplicated).
