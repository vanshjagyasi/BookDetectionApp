"""
app/services/rag_service.py
============================
Stage 2 of the pipeline: retrieve candidate books from ChromaDB.

Uses OpenAI embeddings (via VectorStore) for cosine similarity search.
Returns the top RAG_TOP_K candidates for the LLM to cross-reference.

Input:  VisionExtraction (from vision_service.py)
Output: list of result dicts + formatted context string for the LLM

Dependencies:
  - app.db.vector_store.VectorStore
  - app.schemas.book.VisionExtraction
"""

from __future__ import annotations

from app.db.vector_store import VectorStore
from app.schemas.book import VisionExtraction


class RAGService:
    """
    Retrieves candidate books from ChromaDB using OpenAI embeddings.

    Usage:
        svc = RAGService(vector_store)
        results = svc.retrieve(extraction)
        context = svc.format_context(results)
        # Pass context to LLMService
    """

    def __init__(self, vector_store: VectorStore) -> None:
        """
        Args:
            vector_store: Initialised VectorStore instance (from app.state).
                          Must have initialize() already called.
        """
        self.store = vector_store

    def retrieve(self, extraction: VisionExtraction) -> list[dict]:
        """
        Query ChromaDB for candidate books matching the extraction.

        Step 1: Build a query string from all non-null VisionExtraction fields.
        Step 2: Fetch RAG_TOP_K candidates from ChromaDB.

        Args:
            extraction: VisionExtraction produced by VisionService.

        Returns:
            List of result dicts sorted by relevance (most relevant first).
            Empty list if the collection is empty or no text was extractable.
        """
        title = extraction.visible_title or ""
        author = extraction.visible_author or ""
        rest = " ".join(
            p for p in [extraction.visible_isbn, extraction.other_text] if p and p.strip()
        )

        if author:
            query_text = f"{title} by {author}. {rest}".strip(". ")
        else:
            query_text = " ".join(p for p in [title, rest] if p).strip()

        if not query_text:
            return []

        print(f"[DEBUG] RAG query_text={query_text!r}", flush=True)
        print(f"[DEBUG] RAG db_count={self.store.count()}", flush=True)
        candidates = self.store.query(query_text)
        print(f"[DEBUG] RAG candidates={len(candidates)}", flush=True)

        return candidates

    def format_context(self, results: list[dict]) -> str:
        """
        Format RAG results as a numbered, human-readable context string.

        The formatted string is injected into the LLM prompt so GPT-4o can
        cross-reference the image extraction against known database entries.

        Args:
            results: List of result dicts from retrieve().

        Returns:
            Multi-line string with numbered candidates and their metadata.
            Returns a no-match message if results is empty.
        """
        if not results:
            return (
                "No matching books found in the database. "
                "Identify the book using only what is visible in the image."
            )

        lines = ["Candidate books from database (ranked by relevance):\n"]

        for i, r in enumerate(results, 1):
            m = r["metadata"]
            similarity = round(1 - r["distance"], 3)
            lines.append(
                f"[{i}] Title: {m.get('title', 'Unknown')} | "
                f"Author: {m.get('author', 'Unknown')} | "
                f"ISBN: {m.get('isbn') or 'N/A'} | "
                f"Publisher: {m.get('publisher') or 'N/A'} | "
                f"Year: {m.get('publication_year') or 'N/A'} | "
                f"Genre: {m.get('genre') or 'N/A'} | "
                f"Rating: {m.get('rating') or 0} | "
                f"Similarity: {similarity}\n"
                f"    Synopsis: {(m.get('synopsis') or '')[:200]}\n"
            )

        return "\n".join(lines)
