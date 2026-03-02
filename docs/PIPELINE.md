# Pipeline Deep Dive

This document explains every stage of the book detection pipeline in detail:
what data enters, what leaves, why each design decision was made, and what
happens in edge cases.

---

## Overview

```
HTTP Request (image)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  detection.py  (app/api/v1/routes/detection.py)           │
│  • Validates file type and size                            │
│  • Calls validate_and_encode_image() → base64 string      │
│  • Orchestrates stages 1, 2, 3                             │
└───────────────────────┬───────────────────────────────────┘
                        │
           ┌────────────▼────────────┐
           │    Stage 1: Vision      │
           │    vision_service.py    │
           │                         │
           │  Input:  base64 image   │
           │  Model:  GPT-4o         │
           │  Output: VisionExtraction│
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │    Stage 2: RAG         │
           │    rag_service.py       │
           │                         │
           │  Input:  VisionExtraction│
           │  DB:     ChromaDB        │
           │  Output: list[dict] +   │
           │          context string  │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │    Stage 3: LLM         │
           │    llm_service.py       │
           │                         │
           │  Input:  image +         │
           │          VisionExtraction│
           │          + RAG context   │
           │  Model:  GPT-4o          │
           │  Output: BookInfo        │
           └────────────┬────────────┘
                        │
                        ▼
            HTTP 200 DetectionResponse
```

---

## Stage 1: Vision Extraction

**File:** `app/services/vision_service.py`

### Input
- `image_b64`: Base64-encoded image as a plain string
- `media_type`: MIME type string (`"image/jpeg"`, `"image/png"`, etc.)

### What happens
The image is sent to GPT-4o as a base64 data URI (`data:{media_type};base64,{b64}`)
with `"detail": "high"`. High detail mode splits the image into 512×512px tiles
so small text (ISBN barcodes, author names in small font) is readable.

The system prompt (`VISION_SYSTEM_PROMPT`) instructs GPT-4o to:
1. **Locate the book first** — the image may contain other objects (desk, hands, coffee cup, shelf). GPT-4o focuses exclusively on the book, ignoring all background.
2. Extract only what is **literally visible on the book** — not guess or infer.
3. If multiple books are visible, focus on the most prominent/central one.
4. Return exactly five fields as JSON (enforced by `response_format=json_object`).

### Output
```python
VisionExtraction(
    visible_title="Dune",
    visible_author="Frank Herbert",
    visible_isbn="9780441013593",
    other_text="'The greatest science fiction novel ever written' — NY Times",
    cover_description="Orange desert dunes with silhouetted figure, red/gold colour palette"
)
```

All fields are `Optional[str]` — a spine photo might only have `visible_title`.

### Why GPT-4o (not OCR)?
Traditional OCR (pytesseract, EasyOCR) extracts all text in reading order but:
- Cannot describe cover art (used as supplementary search context)
- Cannot identify which text block is the title vs. a blurb
- Struggles with stylised fonts, rotated text, or text on complex backgrounds

GPT-4o understands the semantic role of each text element, returns structured fields,
AND provides a cover description that improves RAG retrieval for cases where
the title is illegible (e.g., damaged books, unusual typography).

---

## Stage 2: RAG Retrieval

**File:** `app/services/rag_service.py`
**DB:** `app/db/vector_store.py`

### Input
`VisionExtraction` from Stage 1.

### What happens

**Step 2a — Build query string**
Non-null VisionExtraction fields are concatenated into a query string that mirrors
the database document format (`"{title} by {author}. {description}"`):
```
"Dune by Frank Herbert. 'The greatest science fiction novel ever written' — NY Times"
```
The cover description is intentionally excluded — it has no DB equivalent and would
add noise to the embedding. The `other_text` (subtitle, tagline) maps to the
Google Books description slot.

**Step 2b — Bi-encoder: ChromaDB similarity search**
The query string is embedded using `all-MiniLM-L6-v2` and ChromaDB performs
HNSW approximate nearest-neighbour search with cosine distance. Returns
`RAG_FETCH_K` (default: 10) candidates — more than needed so the cross-encoder
has a larger pool to re-rank from.

**Step 2c — Cross-encoder re-ranking**
A CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each
`(query, document)` pair together, seeing both texts simultaneously. This is
~100x more accurate than the bi-encoder but too slow to run on the full
collection. The top `RERANK_TOP_K` (default: 3) results by cross-encoder score
are kept.

If `RERANKER_MODEL` is set to `""` in `.env`, re-ranking is disabled and the
top `RERANK_TOP_K` results in bi-encoder order are used instead.

### Output
```python
[
    {
        "document": "Dune by Frank Herbert. Set in the far future...",
        "metadata": {
            "title": "Dune",
            "author": "Frank Herbert",
            "isbn": "9780441013593",
            "publisher": "Ace",
            "publication_year": 1965,
            "genre": "Science Fiction",
            "rating": 4.5,
            ...
        },
        "distance": 0.082  # cosine distance; lower = more similar
    },
    ...
]
```

These results + a formatted context string are passed to Stage 3.

### What if no books match?
ChromaDB always returns `k` results if the collection has ≥ `k` documents.
There is no "no match" case from ChromaDB itself.

However, if the collection is empty (populate_db.py not run), `VectorStore.query()`
returns `[]`. The LLM then receives "No matching books found in the database" as
its context, and must identify the book from the image alone — resulting in a
lower confidence_score.

### Why cosine similarity?
sentence-transformers produces L2-normalised vectors. For normalised vectors:
- **cosine distance = 1 − cosine similarity** (ranges 0 to 2)
- **Euclidean (L2) distance** also works but is not equivalent to cosine on
  unnormalised vectors and ChromaDB defaults to L2.

We explicitly configure `hnsw:space=cosine` to be correct regardless of
normalisation and to be semantically interpretable (similarity = 1 − distance).

---

## Stage 3: LLM Synthesis

**File:** `app/services/llm_service.py`

### Input
- `extraction`: `VisionExtraction` from Stage 1
- `rag_context`: Formatted string from `RAGService.format_context()`

### What happens
GPT-4o receives a text-only message with two parts:
1. **VisionExtraction JSON** — structured text extracted from the book in Stage 1
2. **RAG context** — the 3 re-ranked database candidates with similarity scores

The system prompt (`LLM_SYSTEM_PROMPT`) instructs GPT-4o to:
- Cross-reference the extracted text with the database candidates
- Select the best matching book (or synthesise if partial match)
- Fill ALL available fields (not just obvious ones)
- Assign a calibrated confidence_score following explicit rules
- Populate ISBN from the database when confidence ≥ 0.90 (even if not visible in image)

### OpenAI Structured Outputs
`client.beta.chat.completions.parse(response_format=BookInfo)` activates
OpenAI's Structured Outputs feature:
- OpenAI converts the `BookInfo` Pydantic schema into a constrained grammar.
- The model can only produce tokens that form valid JSON matching the schema.
- `response.choices[0].message.parsed` returns an already-validated `BookInfo` instance.
- No parsing, no try/except for JSON errors, no schema validation needed.

### Output
```python
BookInfo(
    title="Dune",
    author="Frank Herbert",
    isbn="9780441013593",
    publisher="Ace",
    publication_year=1965,
    genre="Science Fiction",
    tags=["space opera", "political thriller", "ecological sci-fi", "chosen one"],
    synopsis="On the desert planet Arrakis, young Paul Atreides must navigate betrayal...",
    price=9.99,
    rating=4.5,
    confidence_score=0.94
)
```

---

## ChromaDB Schema

**Collection name:** `books` (configurable via `CHROMA_COLLECTION_NAME`)
**Embedding model:** `all-MiniLM-L6-v2` → 384-dimensional vectors
**Distance metric:** cosine

### Document (embedded text)
```
"{title} by {author}. {description}"
```
Example: `"Dune by Frank Herbert. Set on the desert planet Arrakis, ..."`

The description is capped at 1000 characters to keep embedding times consistent.

### Metadata fields (stored verbatim, not embedded)

| Field | Type | Source |
|-------|------|--------|
| `title` | str | Google Books `volumeInfo.title` |
| `author` | str | `volumeInfo.authors` joined with ", " |
| `isbn` | str | `industryIdentifiers` (ISBN-13 preferred) |
| `publisher` | str | `volumeInfo.publisher` |
| `publication_year` | int | First 4 chars of `volumeInfo.publishedDate` |
| `genre` | str | First item of `volumeInfo.categories` |
| `tags` | str | `categories` joined with ", " |
| `synopsis` | str | `volumeInfo.description` capped at 500 chars |
| `price` | float | `saleInfo.retailPrice.amount` or 0.0 |
| `rating` | float | `volumeInfo.averageRating` or 0.0 |

### Document ID
`isbn_13` if available, otherwise `md5(title + first_author)`.
Using ISBN as ID ensures re-running `populate_db.py` updates rather than duplicates.
