"""
scripts/populate_db.py
======================
One-time database seeding script.

Fetches book data from the Google Books API across 12 genre queries and
populates the ChromaDB vector store with ~100 unique books.

Run this ONCE before starting the API server:
    python scripts/populate_db.py

It is safe to re-run — ChromaDB upsert is idempotent (deduplicates by book_id).

Requirements:
    - GOOGLE_BOOKS_API_KEY must be set in .env
    - OPENAI_API_KEY must be set in .env (used for embedding generation)
    - ChromaDB and openai must be installed

What it does:
    1. Issues 12 Google Books API searches across diverse genres.
    2. Parses each result: title, author, ISBN-13, publisher, year,
       categories, description, averageRating.
    3. Builds a rich "document" string for embedding:
           "{title} by {authors}. {description}"
    4. Deduplicates by ISBN-13 (or md5(title+author) if no ISBN).
    5. Upserts each book into ChromaDB with all metadata stored alongside.

After running you should see output like:
    Fetching: best fiction novels  →  added 9 books  (total: 9)
    ...
    Done. 97 unique books in ChromaDB.
"""

import sys
import hashlib
import asyncio
from pathlib import Path

import requests

# Allow running as a script from repo root: python scripts/populate_db.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.db.vector_store import VectorStore


# ------------------------------------------------------------------ #
# Genre queries — broad enough to get diverse coverage               #
# ------------------------------------------------------------------ #
SEARCH_QUERIES = [
    "best fiction novels",
    "classic literature must read",
    "science fiction bestsellers",
    "mystery thriller suspense",
    "biography autobiography memoir",
    "history nonfiction popular",
    "self help personal development",
    "epic fantasy adventure",
    "romance bestsellers",
    "python programming software",
    "machine learning artificial intelligence",
    "young adult coming of age",
    "Atomic Habits",
    "Bitcoins",
    "Blockchains",
    "Finance"
]


def fetch_books_for_query(query: str, api_key: str, max_results: int = 40) -> list[dict]:
    """
    Fetch up to max_results books from Google Books API matching query.

    Args:
        query:       Search string passed to the Google Books volumes endpoint.
        api_key:     Google Books API key.
        max_results: Max items per request (Google Books cap is 40).

    Returns:
        List of raw Google Books "items" dicts.

    Raises:
        requests.HTTPError: On non-200 API response.
    """
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "maxResults": max_results,
        "printType": "books",
        "langRestrict": "en",
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("items", [])


def parse_book(item: dict) -> dict | None:
    """
    Convert a Google Books API item into a VectorStore-ready record.

    Args:
        item: A single dict from the Google Books API "items" array.

    Returns:
        Dict with keys: id, document, metadata.
        Returns None if the item lacks both title and author (unusable).
    """
    info = item.get("volumeInfo", {})
    title = (info.get("title") or "").strip()
    authors = info.get("authors") or []

    if not title or not authors:
        return None

    # Resolve best available ISBN
    isbn_13 = None
    isbn_10 = None
    for identifier in info.get("industryIdentifiers", []):
        if identifier["type"] == "ISBN_13":
            isbn_13 = identifier["identifier"]
        elif identifier["type"] == "ISBN_10":
            isbn_10 = identifier["identifier"]
    isbn = isbn_13 or isbn_10 or ""

    categories = info.get("categories") or []
    description = (info.get("description") or "")[:1000]  # cap to avoid huge embeddings

    # Rich document string — this is what gets embedded for similarity search.
    # Include as much descriptive text as available.
    document = f"{title} by {', '.join(authors)}. {description}".strip()

    # Metadata is stored verbatim in ChromaDB alongside the embedding.
    # All values must be str, int, float, or bool (ChromaDB constraint).
    pub_date = info.get("publishedDate") or ""
    try:
        pub_year = int(pub_date[:4]) if len(pub_date) >= 4 else 0
    except ValueError:
        pub_year = 0

    sale_info = item.get("saleInfo", {}) or {}
    retail = sale_info.get("retailPrice") or {}
    price = float(retail.get("amount") or 0.0)

    metadata = {
        "title": title,
        "author": ", ".join(authors),
        "isbn": isbn,
        "publisher": info.get("publisher") or "",
        "publication_year": pub_year,
        "genre": categories[0] if categories else "",
        "tags": ", ".join(categories),          # stored as comma-separated string
        "synopsis": description[:500],
        "price": price,
        "rating": float(info.get("averageRating") or 0.0),
    }

    # Stable ID: ISBN-13 is best; fall back to hash of title+first author.
    book_id = isbn_13 or hashlib.md5(f"{title}{authors[0]}".encode()).hexdigest()

    return {"id": book_id, "document": document, "metadata": metadata}


async def main() -> None:
    """Seed the ChromaDB collection with books from Google Books API."""
    settings = get_settings()

    if not settings.GOOGLE_BOOKS_API_KEY:
        print("ERROR: GOOGLE_BOOKS_API_KEY is not set in .env")
        print("Get a key at https://console.cloud.google.com/ → Enable 'Books API'")
        sys.exit(1)

    store = VectorStore(settings)
    await store.initialize()
    print(f"ChromaDB initialised at: {settings.CHROMA_PERSIST_DIR}")
    print(f"Collection: {settings.CHROMA_COLLECTION_NAME} (currently {store.count()} books)\n")

    seen_ids: set[str] = set()
    total = 0

    for query in SEARCH_QUERIES:
        print(f"Fetching: {query!r}", end="  →  ", flush=True)
        try:
            items = fetch_books_for_query(query, settings.GOOGLE_BOOKS_API_KEY)
        except requests.HTTPError as exc:
            print(f"SKIPPED (API error: {exc})")
            continue

        added = 0
        for item in items:
            book = parse_book(item)
            if book and book["id"] not in seen_ids:
                store.upsert(book["id"], book["document"], book["metadata"])
                seen_ids.add(book["id"])
                total += 1
                added += 1

        print(f"added {added} books  (total: {total})")

    print(f"\nDone. {total} unique books stored in ChromaDB.")
    print(f"Collection now contains {store.count()} books total.")
    print("\nNext step: uvicorn app.main:app --reload")


if __name__ == "__main__":
    asyncio.run(main())
