"""
app/schemas/errors.py
=====================
Standardised error response models for all 4xx/5xx API responses.

FastAPI will serialise these via HTTPException + custom exception handlers
defined in app/main.py. Having a typed schema means clients always receive
a consistent JSON error envelope regardless of the error source.

Error envelope shape:
    {
        "error": {
            "code": "UNSUPPORTED_MEDIA_TYPE",
            "message": "Unsupported image type: image/bmp. Use JPEG, PNG, or WEBP.",
            "detail": null
        }
    }
"""

from typing import Optional
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """
    Inner error payload describing what went wrong.

    Attributes:
        code:    Machine-readable error code (UPPER_SNAKE_CASE).
        message: Human-readable description of the error.
        detail:  Optional extra context (e.g. field name that failed validation).
    """

    code: str
    message: str
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    """
    Top-level HTTP error response envelope.

    All 4xx and 5xx responses from this API use this shape.

    Example JSON:
        {
            "error": {
                "code": "IMAGE_TOO_LARGE",
                "message": "Image exceeds 20MB limit.",
                "detail": "Received 23.4MB"
            }
        }
    """

    error: ErrorDetail
