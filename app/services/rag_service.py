"""
app/services/rag_service.py
============================
Stage 2 of the pipeline: retrieve and re-rank candidate books from ChromaDB.

Two-stage retrieval pipeline:
  Stage A — Bi-encoder (fast): ChromaDB cosine search fetches RAG_FETCH_K
            candidates (default 10) using the all-MiniLM-L6-v2 sentence
            transformer. Fast enough to search the full collection in ~5ms.

  Stage B — Cross-encoder (precise): CrossEncoder.predict() scores each
            (query, document) pair together, seeing both texts simultaneously.
            Much more accurate than the bi-encoder, but only runs on the 10
            pre-filtered candidates (not the full collection). Returns the
            top RERANK_TOP_K (default 3) by cross-encoder score.

Why two stages?
  A cross-encoder cannot scan all N documents at query time — it's O(N) and
  ~100x slower per comparison than the bi-encoder's approximate nearest-
  neighbour index. The bi-encoder quickly narrows 100+ books down to 10;
  the cross-encoder re-orders those 10 precisely. This pattern appears in
  every production RAG architecture (Cohere Rerank, Pinecone rerank, etc.).

Input:  VisionExtraction (from vision_service.py)
Output: list of re-ranked result dicts + formatted context string for the LLM

Graceful degradation:
  If reranker=None (RERANKER_MODEL not set, or in test environments), the
  service falls back to the bi-encoder order. No code changes required in
  callers.

Dependencies:
  - app.db.vector_store.VectorStore
  - app.schemas.book.VisionExtraction
  - sentence_transformers.CrossEncoder (optional — None disables re-ranking)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

#from langfuse import get_client, observe

from app.db.vector_store import VectorStore
from app.schemas.book import VisionExtraction

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class RAGService:
    """
    Retrieves and re-ranks candidate books using a two-stage pipeline.

    Stage A: bi-encoder (ChromaDB) fetches RAG_FETCH_K candidates fast.
    Stage B: cross-encoder re-ranks them precisely, returns RERANK_TOP_K.

    If reranker is None, returns the top RERANK_TOP_K results in bi-encoder
    order (same behaviour as before re-ranking was added).

    Usage:
        svc = RAGService(vector_store, reranker)
        results = svc.retrieve(extraction)
        context = svc.format_context(results)
        # Pass context to LLMService
    """

    def __init__(
        self,
        vector_store: VectorStore,
        reranker: CrossEncoder | None = None,
    ) -> None:
        """
        Args:
            vector_store: Initialised VectorStore instance (from app.state).
                          Must have initialize() already called.
            reranker:     Optional CrossEncoder for re-ranking. When None,
                          falls back to bi-encoder ordering. Injected from
                          app.state.reranker via get_reranker() dependency.
        """
        self.store = vector_store
        self.reranker = reranker

    #@observe(name="rag-retrieval")
    def retrieve(self, extraction: VisionExtraction) -> list[dict]:
        """
        Query ChromaDB and optionally re-rank results with a cross-encoder.

        Step 1: Build a query string from all non-null VisionExtraction fields.
        Step 2: Fetch RAG_FETCH_K candidates from ChromaDB (bi-encoder).
        Step 3: If reranker is set, re-rank and return top RERANK_TOP_K.
                Otherwise return top RERANK_TOP_K in bi-encoder order.

        Args:
            extraction: VisionExtraction produced by VisionService.

        Returns:
            List of result dicts sorted by relevance (most relevant first):
                [
                    {
                        "document": str,    # the embedded book text
                        "metadata": dict,   # stored book fields
                        "distance": float,  # cosine distance (bi-encoder)
                    },
                    ...
                ]
            Empty list if the collection is empty or no text was extractable.
        """
        # Mirror the DB document format: "{title} by {author}. {description}"
        # (see scripts/populate_db.py → parse_book).  Structural similarity
        # matters for sentence-transformer mean-pooling — a keyword dump like
        # "Title Title Author Author subtitle…" embeds very differently from the
        # natural-sentence format stored in ChromaDB, hurting cosine similarity.
        # other_text (subtitle, tagline) maps well to the Google Books description
        # slot.  cover_description (visual) has no DB equivalent so it's omitted.
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

        fetch_k = self.store.settings.RAG_FETCH_K
        print(f"[DEBUG] RAG query_text={query_text!r}", flush=True)
        print(f"[DEBUG] RAG db_count={self.store.count()}, fetch_k={fetch_k}", flush=True)
        candidates = self.store.query(query_text, n_results=fetch_k)
        print(f"[DEBUG] RAG candidates={len(candidates)}", flush=True)

        if not candidates:
            return []

        results = self._rerank(query_text, candidates)
        print(f"[DEBUG] RAG after rerank={len(results)}", flush=True)

        # Log retrieval metadata as a Langfuse span attribute so it appears
        # in the trace dashboard alongside latency and token counts.
        #top_similarity = round(1 - results[0]["distance"], 3) if results else 0.0
        #get_client().update_current_span(
        #    metadata={
        #        "candidates_fetched": len(candidates),
        #        "candidates_returned": len(results),
        #        "reranked": self.reranker is not None,
        #        "top_similarity": top_similarity,
        #        "query_length": len(query_text),
        #    }
        #)
        return results

    def _rerank(self, query_text: str, results: list[dict]) -> list[dict]:
        """
        Re-rank results with the cross-encoder, or truncate to RERANK_TOP_K.

        If self.reranker is None (re-ranking disabled), returns the first
        RERANK_TOP_K results in their existing bi-encoder order.

        Args:
            query_text: The concatenated VisionExtraction query string.
            results:    Candidates from ChromaDB (bi-encoder order).

        Returns:
            Top RERANK_TOP_K results ordered by cross-encoder score (desc),
            or top RERANK_TOP_K in bi-encoder order if reranker is None.
        """
        top_k = self.store.settings.RERANK_TOP_K

        if self.reranker is None:
            return results[:top_k]

        # CrossEncoder.predict takes a list of (query, document) pairs and
        # returns a float relevance score for each. Higher = more relevant.
        pairs = [(query_text, r["document"]) for r in results]
        scores = self.reranker.predict(pairs)

        # Sort by cross-encoder score descending, keep top-k
        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [r for _, r in ranked[:top_k]]

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

        rank_note = " (cross-encoder re-ranked)" if self.reranker else ""
        lines = [f"Candidate books from database{rank_note} (ranked by relevance):\n"]

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
                f"Bi-encoder similarity: {similarity}\n"
                f"    Synopsis: {(m.get('synopsis') or '')[:200]}\n"
            )

        return "\n".join(lines)
