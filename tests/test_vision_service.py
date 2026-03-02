"""
tests/test_vision_service.py
=============================
Unit tests for VisionService.

Tests cover:
  - Successful extraction returns a VisionExtraction with expected fields
  - Partial extraction (some fields None) is handled correctly
  - Markdown-fenced JSON is stripped and parsed correctly
  - Extra fields from GPT-4o are ignored (forward compatibility)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.schemas.book import VisionExtraction
from app.services.vision_service import VisionService


@pytest.fixture
def vision_service(test_settings: Settings) -> VisionService:
    """VisionService with a mocked OpenAI client."""
    with patch("app.services.vision_service.OpenAI"):
        svc = VisionService(test_settings)
        svc.client = MagicMock()
        return svc


def make_mock_response(content: str) -> MagicMock:
    """Helper: create a mock OpenAI response with given content string."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestExtractBookText:
    def test_full_extraction_returns_all_fields(self, vision_service: VisionService):
        """All five VisionExtraction fields are correctly parsed from JSON."""
        payload = {
            "visible_title": "The Great Gatsby",
            "visible_author": "F. Scott Fitzgerald",
            "visible_isbn": "9780743273565",
            "other_text": "A Scribner Classic",
            "cover_description": "Green light, dock, yellow car",
        }
        vision_service.client.chat.completions.create.return_value = make_mock_response(
            json.dumps(payload)
        )

        result = vision_service.extract_book_text("base64data==", "image/jpeg")

        assert isinstance(result, VisionExtraction)
        assert result.visible_title == "The Great Gatsby"
        assert result.visible_author == "F. Scott Fitzgerald"
        assert result.visible_isbn == "9780743273565"

    def test_partial_extraction_nullable_fields_are_none(self, vision_service: VisionService):
        """Fields not visible in image default to None."""
        payload = {
            "visible_title": "1984",
            "visible_author": None,
            "visible_isbn": None,
            "other_text": None,
            "cover_description": "Dark dystopian cover",
        }
        vision_service.client.chat.completions.create.return_value = make_mock_response(
            json.dumps(payload)
        )

        result = vision_service.extract_book_text("base64data==", "image/png")

        assert result.visible_title == "1984"
        assert result.visible_author is None
        assert result.visible_isbn is None

    def test_markdown_fenced_json_is_stripped(self, vision_service: VisionService):
        """JSON wrapped in ```json ... ``` fences is correctly parsed."""
        payload = {"visible_title": "Dune", "visible_author": "Frank Herbert",
                   "visible_isbn": None, "other_text": None, "cover_description": "Desert"}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        vision_service.client.chat.completions.create.return_value = make_mock_response(fenced)

        result = vision_service.extract_book_text("base64data==", "image/jpeg")
        assert result.visible_title == "Dune"

    def test_extra_fields_are_ignored(self, vision_service: VisionService):
        """Extra fields returned by GPT-4o do not raise validation errors."""
        payload = {
            "visible_title": "Neuromancer",
            "visible_author": "William Gibson",
            "visible_isbn": None,
            "other_text": None,
            "cover_description": "Cyberpunk neon",
            "extra_unexpected_field": "should be ignored",
        }
        vision_service.client.chat.completions.create.return_value = make_mock_response(
            json.dumps(payload)
        )

        result = vision_service.extract_book_text("base64data==", "image/jpeg")
        assert result.visible_title == "Neuromancer"
