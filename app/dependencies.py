"""
app/dependencies.py
====================
FastAPI dependency injection helpers.

The VectorStore is stored once on app.state during startup (see main.py
lifespan) and exposed here as a dependency function for use with Depends().

Usage in a route:
    from app.dependencies import get_vector_store

    @router.post("/detect-book")
    async def detect_book(
        vector_store: VectorStore = Depends(get_vector_store),
    ):
        rag_svc = RAGService(vector_store)
        ...
"""

from fastapi import Request

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
