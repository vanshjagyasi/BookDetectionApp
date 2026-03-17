"""
app/main.py
===========
FastAPI application factory and lifespan manager.

This is the entry point for the web server:
    uvicorn app.main:app --reload

Application startup (lifespan):
  1. Instantiate VectorStore with app settings.
  2. Call vector_store.initialize() — connects to ChromaDB on disk.
  3. Store the initialised VectorStore on app.state so route handlers can
     access it via FastAPI's dependency injection (see app/dependencies.py).

IMPORTANT — Architecture invariant:
  The VectorStore (containing the ChromaDB client) is intentionally
  initialised exactly ONCE here. Never create a VectorStore inside a
  route handler.

Registered routes:
  POST /api/v1/detect-book  →  app/api/v1/routes/detection.py

Swagger UI available at:
  http://localhost:8000/docs  (when running with uvicorn)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routes.detection import router as detection_router
from app.config import get_settings
from app.db.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.

    Startup:
      - Initialise VectorStore (connects to ChromaDB).
      - Store on app.state for dependency injection.

    Shutdown:
      - Nothing to do — ChromaDB PersistentClient flushes on its own.
    """
    settings = get_settings()

    # --- VectorStore (ChromaDB + OpenAI embeddings) ---
    vector_store = VectorStore(settings)
    await vector_store.initialize()
    app.state.vector_store = vector_store

    book_count = vector_store.count()
    if book_count == 0:
        print(
            "\n[WARNING] ChromaDB collection is empty. "
            "Run: python scripts/populate_db.py\n"
        )
    else:
        print(f"\n[OK] VectorStore ready — {book_count} books in ChromaDB.\n")

    yield
    # Shutdown: nothing explicit needed; ChromaDB persists on disk automatically.


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI app.

    Using a factory function (instead of a module-level `app = FastAPI()`)
    makes the app easier to test: tests can call create_app() to get a
    fresh instance with mocked dependencies.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Book Detection API",
        description=(
            "RAG-powered book identification from cover images. "
            "Upload a photo of any book cover (front, back, or side) and receive "
            "structured JSON metadata identified via GPT-4o vision + ChromaDB vector search."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allows the frontend to call the API from a browser.
    # Restrict ALLOWED_ORIGINS in production (e.g. your frontend domain).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Register the detection router under /api/v1
    app.include_router(detection_router, prefix="/api/v1")

    # Health check endpoint — used by Docker healthcheck and load balancers
    @app.get("/health", tags=["Health"], summary="API health check")
    async def health() -> dict:
        """Returns 200 OK with book count when the API is ready."""
        store: VectorStore = app.state.vector_store
        return {
            "status": "ok",
            "books_in_db": store.count(),
            "model": settings.OPENAI_MODEL,
        }

    # Global exception handler for unexpected errors
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred.",
                    "detail": str(exc),
                }
            },
        )

    return app


# Module-level app instance — referenced by uvicorn as "app.main:app"
app = create_app()
