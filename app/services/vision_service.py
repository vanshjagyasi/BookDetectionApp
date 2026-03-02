"""
app/services/vision_service.py
================================
Stage 1 of the pipeline: extract visible text from a book cover image.

Uses GPT-4o's multimodal capability to read text and describe the visual
content of a book cover (front, back, or side/spine).

Input:  base64-encoded image bytes + MIME type string
Output: VisionExtraction Pydantic model

Key design decisions:
  - response_format={"type": "json_object"} forces GPT-4o to return valid JSON.
    This is more reliable than asking it to "respond in JSON" in a plain prompt.
  - "detail": "high" tells OpenAI to use the full 2048-tile resolution for
    reading small text (ISBN numbers, fine print).
  - The system prompt is strict about not inventing information — only what is
    literally visible in the image.

To swap this service for a different vision provider:
  See docs/EXTENDING.md → "Swapping the Vision Provider".

Dependencies:
  - openai SDK  (pip install openai)
  - app.schemas.book.VisionExtraction
  - app.config.Settings (OPENAI_API_KEY, OPENAI_MODEL)
"""

import json

from langfuse import observe
from langfuse.openai import OpenAI  # Drop-in: auto-traces prompts, responses, tokens

from app.config import Settings
from app.schemas.book import VisionExtraction


VISION_SYSTEM_PROMPT = """You are a precise book cover text extraction system.

IMPORTANT — the image may contain a book alongside other objects (a desk, hands, coffee cup, shelf, etc.). Your FIRST step is to LOCATE the book in the image. Focus exclusively on the book — ignore all background objects, surfaces, and non-book items entirely.

Your task: examine the BOOK in the provided image and extract ALL visible text and visual details FROM THE BOOK ONLY.

Focus on extracting:
- visible_title: the main title text as it appears on the book cover
- visible_author: author name(s) as they appear on the book
- visible_isbn: any ISBN number or barcode digits on the book (usually on back cover or inside flap)
- other_text: subtitle, series name, edition, tagline, publisher name, any other readable text ON THE BOOK
- cover_description: brief description of the book's cover art, colour scheme, imagery, visual style (describe only the book, not the surroundings)

Rules:
1. Only report what is ACTUALLY visible ON THE BOOK — do not invent or guess.
2. Ignore any text or objects in the background that are NOT part of the book.
3. If a field is not visible on the book, set it to null.
4. For visible_isbn: extract the numeric digits only (e.g. "9780743273565").
5. If multiple books are visible, focus on the most prominent/central one.
6. Always return valid JSON with exactly these five keys:
   visible_title, visible_author, visible_isbn, other_text, cover_description
"""


class VisionService:
    """
    Wraps the OpenAI GPT-4o multimodal API for book cover text extraction.

    Sends the image as a base64 data URI with high detail enabled.
    Returns a VisionExtraction with all visible text fields populated.
    Fields default to None if not visible in the image.

    Usage:
        svc = VisionService(settings)
        extraction = svc.extract_book_text(image_b64, "image/jpeg")
        print(extraction.visible_title)
    """

    def __init__(self, settings: Settings) -> None:
        """
        Args:
            settings: Application settings providing OPENAI_API_KEY and OPENAI_MODEL.
        """
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    @observe(name="vision-extraction")
    def extract_book_text(self, image_b64: str, media_type: str) -> VisionExtraction:
        """
        Send a book cover image to GPT-4o and extract visible text fields.

        Args:
            image_b64:  Base64-encoded image bytes as a plain string
                        (NOT a data URI — just the base64 content).
            media_type: MIME type string, e.g. "image/jpeg" or "image/png".
                        Used to construct the data URI sent to OpenAI.

        Returns:
            VisionExtraction with fields populated from what is visible in the image.
            All fields default to None if not detectable.

        Raises:
            openai.APIError: On API call failure, authentication error, or timeout.
            json.JSONDecodeError: If GPT-4o returns malformed JSON despite
                                  response_format=json_object (extremely rare).
            pydantic.ValidationError: If the returned JSON has unexpected field types.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                # OpenAI accepts base64 images as data URIs
                                "url": f"data:{media_type};base64,{image_b64}",
                                # "high" detail: split image into 512px tiles for
                                # better small-text OCR (ISBN numbers, fine print)
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Locate the book in this image, ignore everything else, and extract all text visible on the book. Return as JSON.",
                        },
                    ],
                },
            ],
            max_tokens=1024,
        )

        raw_text = response.choices[0].message.content or "{}"

        # response_format=json_object should prevent markdown fences, but
        # we strip them defensively in case of unexpected formatting.
        if "```" in raw_text:
            raw_text = raw_text.split("```json")[-1].split("```")[0].strip()

        data = json.loads(raw_text)

        # Allow GPT-4o to return extra keys — model_validate with extra="ignore"
        # keeps us forward-compatible if the model returns bonus fields.
        return VisionExtraction.model_validate(data)
