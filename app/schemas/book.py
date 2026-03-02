"""
app/schemas/book.py
===================
Pydantic data models that form the contract between all three pipeline stages
(vision, RAG, LLM) and the REST API response.

Key models:
  VisionExtraction  — what GPT-4o reads from the cover image
  BookInfo          — the final structured JSON returned to the caller
  DetectionResponse — HTTP response wrapper (success flag + BookInfo)

These schemas are the single source of truth for field names and types.
To add a new output field:
  1. Add it to BookInfo below.
  2. Update the LLM system prompt in app/services/llm_service.py to populate it.
  3. The OpenAI Structured Outputs feature will enforce it automatically.
"""

from typing import Optional
from pydantic import BaseModel, Field


class VisionExtraction(BaseModel):
    """
    Text and visual details extracted from a book cover image by GPT-4o.

    All fields are Optional because not every cover shows every piece of
    information (e.g. side spines show only title/author, back covers
    show ISBN/synopsis but not always a visible title block).

    This model is consumed internally by RAGService and LLMService.
    It is NOT part of the public API response.
    """

    visible_title: Optional[str] = Field(
        None, description="Title text read directly from the cover image."
    )
    visible_author: Optional[str] = Field(
        None, description="Author name(s) read directly from the cover image."
    )
    visible_isbn: Optional[str] = Field(
        None,
        description="ISBN barcode digits visible on the cover (usually on back cover).",
    )
    other_text: Optional[str] = Field(
        None,
        description="Any other readable text: subtitle, tagline, edition, series info.",
    )
    cover_description: Optional[str] = Field(
        None,
        description="Visual description of the cover art, colour palette, style, mood.",
    )


class BookInfo(BaseModel):
    """
    Fully resolved book metadata returned to the API caller.

    Produced by LLMService by cross-referencing VisionExtraction with
    ChromaDB RAG candidates. Fields are populated from the best available
    source: database record (if matched) or LLM inference from the image.

    confidence_score is the only required field — all others may be None
    if the book cannot be identified or the data is unavailable.
    """

    title: Optional[str] = Field(None, description="Full book title.")
    author: Optional[str] = Field(None, description="Primary author name(s).")
    isbn: Optional[str] = Field(
        None, description="ISBN-13 preferred; ISBN-10 as fallback."
    )
    publisher: Optional[str] = Field(None, description="Publishing house name.")
    publication_year: Optional[int] = Field(
        None, ge=1000, le=2200, description="Year of publication (4-digit integer)."
    )
    genre: Optional[str] = Field(
        None, description="Primary genre classification (e.g. 'Science Fiction')."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Descriptive tags / sub-genres (e.g. ['dystopia', 'space opera']).",
    )
    synopsis: Optional[str] = Field(
        None, description="Brief summary of the book's content or plot."
    )
    price: Optional[float] = Field(
        None, ge=0, description="Typical retail price in USD (from Google Books data)."
    )
    rating: Optional[float] = Field(
        None, ge=0, le=5, description="Average reader rating out of 5."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Model confidence in the identification accuracy. "
            "≥0.9 = ISBN confirmed; 0.7–0.9 = title+author match; "
            "0.4–0.7 = partial match; <0.4 = uncertain inference."
        ),
    )


class DetectionResponse(BaseModel):
    """
    Top-level HTTP response returned by POST /api/v1/detect-book.

    Attributes:
        success: Always True on HTTP 200. Error cases return 4xx/5xx instead.
        book: The resolved BookInfo object.
        extraction_notes: Optional human-readable summary of what the vision
                          step detected and how many RAG candidates matched.
                          Useful for debugging and logging.
    """

    success: bool = Field(True, description="True on successful identification.")
    book: BookInfo
    extraction_notes: Optional[str] = Field(
        None,
        description="Debug info: what vision extracted and RAG match count.",
    )
