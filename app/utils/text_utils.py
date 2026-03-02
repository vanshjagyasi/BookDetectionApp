"""
app/utils/text_utils.py
========================
Lightweight text cleaning helpers used across the pipeline.

Currently used by:
  - scripts/populate_db.py (clean description text before embedding)
  - Potentially by rag_service.py for query normalisation

Kept minimal intentionally — only add utilities that are actually reused
in two or more places.
"""

import re


def clean_text(text: str) -> str:
    """
    Normalise whitespace and remove control characters from a string.

    Args:
        text: Any input string.

    Returns:
        String with leading/trailing whitespace stripped, internal runs of
        whitespace collapsed to a single space, and control characters removed.

    Examples:
        clean_text("  Hello\\n\\nWorld  ") → "Hello World"
        clean_text("Dune\\t\\tby Frank") → "Dune by Frank"
    """
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)   # remove control chars
    text = re.sub(r"\s+", " ", text)                 # collapse whitespace
    return text.strip()


def truncate(text: str, max_chars: int) -> str:
    """
    Truncate text to max_chars characters, adding "…" if truncated.

    Args:
        text:      Input string.
        max_chars: Maximum number of characters in the output.

    Returns:
        Original string if len <= max_chars, otherwise truncated with "…".
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
