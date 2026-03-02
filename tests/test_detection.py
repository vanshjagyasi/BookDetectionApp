"""
tests/test_detection.py
========================
Integration tests for the POST /api/v1/detect-book endpoint.

Tests cover:
  - Happy path: valid JPEG returns BookInfo JSON
  - Unsupported file type returns 415
  - Oversized file returns 413
  - Corrupt image bytes return 422
  - Health check endpoint

All tests use mocked OpenAI and ChromaDB — no real API calls are made.
"""

import pytest
from fastapi.testclient import TestClient


class TestDetectBook:
    """Tests for POST /api/v1/detect-book."""

    def test_valid_jpeg_returns_book_info(self, client: TestClient, sample_image_bytes: bytes):
        """Happy path: a valid JPEG image returns a 200 with BookInfo."""
        response = client.post(
            "/api/v1/detect-book",
            files={"file": ("cover.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "book" in data

        book = data["book"]
        assert book["title"] == "Dune"
        assert book["author"] == "Frank Herbert"
        assert book["isbn"] == "9780441013593"
        assert isinstance(book["confidence_score"], float)
        assert 0.0 <= book["confidence_score"] <= 1.0

    def test_response_includes_extraction_notes(self, client: TestClient, sample_image_bytes: bytes):
        """extraction_notes field is present and non-empty on success."""
        response = client.post(
            "/api/v1/detect-book",
            files={"file": ("cover.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("extraction_notes") is not None

    def test_png_image_accepted(self, client: TestClient):
        """PNG images are accepted (not just JPEG)."""
        from io import BytesIO
        from PIL import Image

        img = Image.new("RGB", (100, 150), color=(0, 100, 200))
        buf = BytesIO()
        img.save(buf, format="PNG")

        response = client.post(
            "/api/v1/detect-book",
            files={"file": ("cover.png", buf.getvalue(), "image/png")},
        )
        assert response.status_code == 200

    def test_unsupported_type_returns_415(self, client: TestClient):
        """BMP and other unsupported formats return 415."""
        response = client.post(
            "/api/v1/detect-book",
            files={"file": ("cover.bmp", b"fake", "image/bmp")},
        )
        assert response.status_code == 415
        assert "Unsupported" in response.json()["detail"]

    def test_oversized_image_returns_413(self, client: TestClient):
        """Images exceeding MAX_IMAGE_SIZE_MB (5MB in test settings) return 413."""
        # 6MB of fake bytes
        big_bytes = b"x" * (6 * 1024 * 1024)
        response = client.post(
            "/api/v1/detect-book",
            files={"file": ("big.jpg", big_bytes, "image/jpeg")},
        )
        assert response.status_code == 413

    def test_corrupt_image_returns_422(self, client: TestClient):
        """Random bytes that aren't a valid image return 422."""
        response = client.post(
            "/api/v1/detect-book",
            files={"file": ("corrupt.jpg", b"not-an-image-at-all", "image/jpeg")},
        )
        assert response.status_code == 422

    def test_no_file_returns_422(self, client: TestClient):
        """Missing file field returns a 422 validation error."""
        response = client.post("/api/v1/detect-book")
        assert response.status_code == 422


class TestHealth:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client: TestClient):
        """Health endpoint returns 200 with expected fields."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert isinstance(data["books_in_db"], int)
        assert data["books_in_db"] >= 0
        assert "model" in data
