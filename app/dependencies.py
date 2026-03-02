"""
app/dependencies.py
====================
FastAPI dependency injection helpers.

FastAPI's Depends() system is used to inject shared resources (like the
VectorStore and CrossEncoder re-ranker) into route handlers without passing
them through global state or re-creating them per request.

Both resources are stored once on app.state during startup (see main.py
lifespan) and exposed here as dependency functions for use with Depends().

Usage in a route:
    from app.dependencies import get_vector_store, get_reranker

    @router.post("/detect-book")
    async def detect_book(
        vector_store: VectorStore = Depends(get_vector_store),
        reranker = Depends(get_reranker),
    ):
        rag_svc = RAGService(vector_store, reranker)
        ...
"""

from fastapi import Request
from sentence_transformers import CrossEncoder

from app.db.vector_store import VectorStore


def get_vector_store(request: Request) -> VectorStore:
    """
    Retrieve the shared VectorStore instance from application state.

    The VectorStore is initialised once in main.py's lifespan() and stored
    at app.state.vector_store. This dependency function exposes it for
    injection into route handlers via FastAPI's Depends() system.

    Args:
        request: The current FastAPI Request (injected automatically by FastAPI).

    Returns:
        The application-wide VectorStore instance (already initialised).
    """
    return request.app.state.vector_store


def get_reranker(request: Request) -> CrossEncoder | None:
    """
    Retrieve the shared CrossEncoder re-ranker instance from application state.

    The CrossEncoder is loaded once in main.py's lifespan() and stored at
    app.state.reranker. Returns None if RERANKER_MODEL was not configured,
    in which case RAGService falls back to bi-encoder ordering.

    Args:
        request: The current FastAPI Request (injected automatically by FastAPI).

    Returns:
        CrossEncoder instance, or None if re-ranking is disabled.
    """
    return request.app.state.reranker
