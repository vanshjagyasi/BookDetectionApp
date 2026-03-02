"""
scripts/verify_db.py
====================
Inspect the contents of the ChromaDB collection.

Usage:
    python scripts/verify_db.py              # print first 10 books
    python scripts/verify_db.py --all        # print all books
    python scripts/verify_db.py --query "dune"  # run a semantic search

Useful for:
  - Verifying that populate_db.py ran successfully.
  - Checking metadata quality.
  - Manually testing vector similarity search.
  - Debugging RAG retrieval issues.
"""

import sys
import asyncio
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.db.vector_store import VectorStore


async def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ChromaDB book collection.")
    parser.add_argument("--all", action="store_true", help="Print all stored books.")
    parser.add_argument("--query", type=str, default="", help="Run a semantic search.")
    parser.add_argument("--limit", type=int, default=10, help="Max books to print.")
    args = parser.parse_args()

    settings = get_settings()
    store = VectorStore(settings)
    await store.initialize()

    total = store.count()
    print(f"Collection: {settings.CHROMA_COLLECTION_NAME!r}")
    print(f"Total books: {total}")
    print(f"ChromaDB path: {settings.CHROMA_PERSIST_DIR}\n")

    if total == 0:
        print("Collection is empty. Run: python scripts/populate_db.py")
        return

    if args.query:
        # Semantic search mode
        print(f"Searching for: {args.query!r}\n")
        results = store.query(args.query, n_results=min(args.limit, total))
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            similarity = 1 - r["distance"]
            print(
                f"[{i}] {m.get('title', '?')} by {m.get('author', '?')}\n"
                f"    ISBN: {m.get('isbn') or 'N/A'} | "
                f"Year: {m.get('publication_year') or 'N/A'} | "
                f"Genre: {m.get('genre') or 'N/A'} | "
                f"Similarity: {similarity:.3f}\n"
                f"    {m.get('synopsis', '')[:120]}...\n"
            )
        return

    # Listing mode — peek at raw collection data
    limit = total if args.all else min(args.limit, total)
    print(f"Showing {limit} of {total} books:\n")

    raw = store.collection.get(
        limit=limit,
        include=["metadatas", "documents"],
    )
    for i, (meta, doc) in enumerate(zip(raw["metadatas"], raw["documents"]), 1):
        print(
            f"[{i}] {meta.get('title', '?')} — {meta.get('author', '?')}\n"
            f"    ISBN: {meta.get('isbn') or 'N/A'} | "
            f"{meta.get('publication_year') or 'N/A'} | "
            f"{meta.get('genre') or 'N/A'} | "
            f"★ {meta.get('rating') or 0}\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
