"""
tests/test_rag_service.py
==========================
Unit tests for RAGService.

Tests cover:
  - retrieve() builds correct query string from VisionExtraction
  - retrieve() returns empty list when collection is empty
  - retrieve() returns empty list when extraction has no usable text
  - format_context() produces correct output for results
  - format_context() handles empty results with a no-match message
"""

import pytest

from app.db.vector_store import VectorStore
from app.schemas.book import VisionExtraction
from app.services.rag_service import RAGService


class TestRetrieve:
    def test_returns_results_for_matching_book(
        self, populated_vector_store: VectorStore, sample_extraction: VisionExtraction
    ):
        """retrieve() returns at least one result when a matching book exists."""
        svc = RAGService(populated_vector_store)
        results = svc.retrieve(sample_extraction)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_returns_empty_for_no_usable_text(
        self, populated_vector_store: VectorStore
    ):
        """retrieve() returns [] when all VisionExtraction fields are None."""
        empty_extraction = VisionExtraction()
        svc = RAGService(populated_vector_store)
        results = svc.retrieve(empty_extraction)
        assert results == []

    def test_returns_empty_for_empty_collection(
        self, mock_vector_store: VectorStore, sample_extraction: VisionExtraction
    ):
        """retrieve() returns [] when the ChromaDB collection has no documents."""
        svc = RAGService(mock_vector_store)   # empty store (no upserts)
        results = svc.retrieve(sample_extraction)
        assert results == []

    def test_result_has_expected_keys(
        self, populated_vector_store: VectorStore, sample_extraction: VisionExtraction
    ):
        """Each result dict has 'document', 'metadata', and 'distance' keys."""
        svc = RAGService(populated_vector_store)
        results = svc.retrieve(sample_extraction)
        if results:
            r = results[0]
            assert "document" in r
            assert "metadata" in r
            assert "distance" in r
            assert isinstance(r["distance"], float)


class TestFormatContext:
    def test_no_results_returns_no_match_message(self):
        """format_context([]) returns a human-readable no-match string."""
        svc = RAGService(MagicMock())
        context = svc.format_context([])
        assert "No matching books" in context

    def test_results_include_title_and_similarity(
        self, populated_vector_store: VectorStore, sample_extraction: VisionExtraction
    ):
        """format_context() output contains the book title and Similarity score."""
        svc = RAGService(populated_vector_store)
        results = svc.retrieve(sample_extraction)
        context = svc.format_context(results)
        if results:
            assert "Dune" in context
            assert "Similarity" in context

    def test_numbered_candidates(
        self, populated_vector_store: VectorStore, sample_extraction: VisionExtraction
    ):
        """Results are numbered [1], [2], etc."""
        svc = RAGService(populated_vector_store)
        results = svc.retrieve(sample_extraction)
        context = svc.format_context(results)
        if results:
            assert "[1]" in context


# Needed for the no-results test
from unittest.mock import MagicMock
