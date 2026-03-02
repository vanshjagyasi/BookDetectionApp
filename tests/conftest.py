"""
tests/conftest.py
=================
Shared pytest fixtures for the test suite.

All fixtures use mocking to avoid real API calls or disk I/O:
  - OpenAI client is replaced with a MagicMock
  - ChromaDB is replaced with an in-memory client
  - SentenceTransformer.encode() is patched to return a fixed vector

This means the full test suite runs without:
  - An OPENAI_API_KEY
  - A GOOGLE_BOOKS_API_KEY
  - ChromaDB data on disk
  - Internet access

Usage in a test file:
    def test_something(client, mock_openai_client, mock_vector_store):
        ...
"""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.db.vector_store import VectorStore
from app.main import create_app
from app.schemas.book import BookInfo, VisionExtraction


# ------------------------------------------------------------------
# Settings fixture — override env vars for testing
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Return a Settings instance with test values (no real API keys needed)."""
    return Settings(
        OPENAI_API_KEY="sk-test-key",
        OPENAI_MODEL="gpt-4o",
        GOOGLE_BOOKS_API_KEY="test-books-key",
        CHROMA_PERSIST_DIR=":memory:",   # not actually used — we mock chromadb
        CHROMA_COLLECTION_NAME="test_books",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        RAG_TOP_K=3,
        MAX_IMAGE_SIZE_MB=5,
        ALLOWED_ORIGINS=["*"],
    )


# ------------------------------------------------------------------
# VectorStore fixture — in-memory ChromaDB + mocked embedding model
# ------------------------------------------------------------------

@pytest.fixture
def mock_vector_store(test_settings) -> VectorStore:
    """
    Return an initialised VectorStore backed by an in-memory ChromaDB.

    The SentenceTransformer model is mocked to return a fixed 384-dim vector.
    This avoids downloading or loading the 80MB model during tests.
    """
    store = VectorStore(test_settings)

    # Use in-memory chromadb client (no disk writes)
    store.client = chromadb.EphemeralClient()
    store.collection = store.client.get_or_create_collection(
        name=test_settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Mock the embedding model — returns a fixed unit vector
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 384
    store.embedding_model = mock_model

    return store


# ------------------------------------------------------------------
# Sample book fixture
# ------------------------------------------------------------------

@pytest.fixture
def sample_book_metadata() -> dict:
    return {
        "title": "Dune",
        "author": "Frank Herbert",
        "isbn": "9780441013593",
        "publisher": "Ace",
        "publication_year": 1965,
        "genre": "Science Fiction",
        "tags": "Science Fiction, Space Opera",
        "synopsis": "Epic sci-fi about a desert planet.",
        "price": 9.99,
        "rating": 4.5,
    }


@pytest.fixture
def populated_vector_store(mock_vector_store, sample_book_metadata) -> VectorStore:
    """VectorStore with one sample book pre-inserted."""
    mock_vector_store.upsert(
        book_id="9780441013593",
        document="Dune by Frank Herbert. Epic science fiction about a desert planet.",
        metadata=sample_book_metadata,
    )
    return mock_vector_store


# ------------------------------------------------------------------
# Mock VisionExtraction and BookInfo
# ------------------------------------------------------------------

@pytest.fixture
def sample_extraction() -> VisionExtraction:
    return VisionExtraction(
        visible_title="Dune",
        visible_author="Frank Herbert",
        visible_isbn="9780441013593",
        other_text="A Swordfish Book",
        cover_description="Orange desert dunes with figure silhouette",
    )


@pytest.fixture
def sample_book_info() -> BookInfo:
    return BookInfo(
        title="Dune",
        author="Frank Herbert",
        isbn="9780441013593",
        publisher="Ace",
        publication_year=1965,
        genre="Science Fiction",
        tags=["space opera", "dystopian"],
        synopsis="On the desert planet Arrakis...",
        price=9.99,
        rating=4.5,
        confidence_score=0.94,
    )


# ------------------------------------------------------------------
# Mock OpenAI client
# ------------------------------------------------------------------

@pytest.fixture
def mock_openai_client(sample_extraction, sample_book_info):
    """
    Mock openai.OpenAI client for both vision and LLM service tests.

    Patches openai.OpenAI so any instantiation in the services returns a mock
    that responds as expected.
    """
    import json

    # Vision response: returns VisionExtraction as JSON
    vision_response = MagicMock()
    vision_response.choices = [MagicMock()]
    vision_response.choices[0].message.content = sample_extraction.model_dump_json()

    # LLM response: returns BookInfo as parsed attribute (Structured Outputs)
    llm_response = MagicMock()
    llm_response.choices = [MagicMock()]
    llm_response.choices[0].message.parsed = sample_book_info

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = vision_response
    mock_client.beta.chat.completions.parse.return_value = llm_response

    with patch("app.services.vision_service.OpenAI", return_value=mock_client), \
         patch("app.services.llm_service.OpenAI", return_value=mock_client):
        yield mock_client


# ------------------------------------------------------------------
# Test image fixture
# ------------------------------------------------------------------

@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a small valid JPEG image in memory (no disk file needed)."""
    img = Image.new("RGB", (200, 300), color=(255, 165, 0))  # orange rectangle
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------
# FastAPI TestClient fixture
# ------------------------------------------------------------------

@pytest.fixture
def client(test_settings, populated_vector_store, mock_openai_client):
    """
    Return a FastAPI TestClient with mocked dependencies.

    The VectorStore is injected via app.state so the lifespan startup
    (which would load real ChromaDB + model) is bypassed.
    """
    app = create_app()

    # Override lifespan-set app state before client starts
    async def mock_lifespan(app):
        app.state.vector_store = populated_vector_store
        yield

    app.router.lifespan_context = mock_lifespan

    with TestClient(app) as c:
        # Manually set app.state for the test client
        app.state.vector_store = populated_vector_store
        yield c
