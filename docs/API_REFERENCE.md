# API Reference

Base URL: `http://localhost:8000`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/detect-book` | Identify a book from its cover image |
| GET | `/health` | API health check and status |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | ReDoc documentation UI |

---

## POST `/api/v1/detect-book`

Identify a book from a cover photo.

### Request

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Book cover image. JPEG, PNG, WEBP, or GIF. Max size: `MAX_IMAGE_SIZE_MB` (default 20MB). |

### Response `200 OK`

```json
{
  "success": true,
  "book": {
    "title": "Dune",
    "author": "Frank Herbert",
    "isbn": "9780441013593",
    "publisher": "Ace",
    "publication_year": 1965,
    "genre": "Science Fiction",
    "tags": ["space opera", "political thriller", "ecological sci-fi"],
    "synopsis": "On the desert planet Arrakis, young nobleman Paul Atreides...",
    "price": 9.99,
    "rating": 4.5,
    "confidence_score": 0.93
  },
  "extraction_notes": "Vision read: 'DUNE' | Author: 'Frank Herbert' | RAG matched 3 candidate(s)"
}
```

### BookInfo Schema

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `title` | string | Yes | Full book title |
| `author` | string | Yes | Primary author name(s) |
| `isbn` | string | Yes | ISBN-13 preferred; ISBN-10 as fallback |
| `publisher` | string | Yes | Publishing house name |
| `publication_year` | integer | Yes | 4-digit year of publication |
| `genre` | string | Yes | Primary genre (e.g. `"Science Fiction"`) |
| `tags` | string[] | No (default `[]`) | Sub-genres and descriptive tags |
| `synopsis` | string | Yes | 1–3 sentence plot/content summary |
| `price` | float | Yes | Retail price in USD |
| `rating` | float | Yes | Average reader rating (0–5) |
| `confidence_score` | float | **No** | Model confidence (0.0–1.0) — always present |

### Confidence Score Guide

| Value | Meaning |
|-------|---------|
| ≥ 0.95 | ISBN visible in image AND matches a database record exactly |
| 0.90–0.94 | Title AND author clearly readable, database candidate matches with good similarity (≥ 0.60) |
| 0.80–0.89 | Title AND author clearly readable, database match with lower similarity |
| 0.70–0.79 | Title AND author clearly readable, even without a database match |
| 0.40–0.69 | Only title OR author readable (partial signal) |
| < 0.40 | No text evidence; identification from visual inference alone |

**ISBN population:** If ISBN is not visible in the image but confidence is ≥ 0.90, the ISBN is populated from the matched database record.

### Error Responses

#### `413 Request Entity Too Large`
Image file exceeds the configured size limit.
```json
{
  "detail": "Image size 23.4MB exceeds the 20MB limit."
}
```

#### `415 Unsupported Media Type`
File content type is not an accepted image format.
```json
{
  "detail": "Unsupported image type: 'image/bmp'. Supported: image/gif, image/jpeg, image/png, image/webp"
}
```

#### `422 Unprocessable Entity`
Image bytes are corrupt or cannot be decoded by Pillow.
```json
{
  "detail": "Invalid image file: cannot identify image file"
}
```

#### `500 Internal Server Error`
OpenAI API failure, ChromaDB error, or unexpected exception.
```json
{
  "error": {
    "code": "INTERNAL_SERVER_ERROR",
    "message": "An unexpected error occurred.",
    "detail": "<exception message>"
  }
}
```

---

## GET `/health`

Returns API status and database statistics.

### Response `200 OK`
```json
{
  "status": "ok",
  "books_in_db": 97,
  "model": "gpt-4o"
}
```

If ChromaDB is empty (populate_db.py not yet run), `books_in_db` will be `0`
and the API will still function but with lower identification accuracy.

---

## curl Examples

### Basic book detection
```bash
curl -X POST http://localhost:8000/api/v1/detect-book \
     -F "file=@front_cover.jpg"
```

### With verbose output (see headers and status)
```bash
curl -v -X POST http://localhost:8000/api/v1/detect-book \
     -F "file=@back_cover.png"
```

### Save response to file
```bash
curl -X POST http://localhost:8000/api/v1/detect-book \
     -F "file=@spine.jpg" \
     -o result.json
```

### Health check
```bash
curl http://localhost:8000/health
```

### Python (requests)
```python
import requests

with open("book_cover.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/detect-book",
        files={"file": ("book_cover.jpg", f, "image/jpeg")},
    )

data = response.json()
book = data["book"]
print(f"{book['title']} by {book['author']}  (confidence: {book['confidence_score']:.0%})")
```

### JavaScript (fetch)
```javascript
const formData = new FormData();
formData.append("file", imageFile);  // imageFile is a File object

const response = await fetch("http://localhost:8000/api/v1/detect-book", {
  method: "POST",
  body: formData,
});

const { book } = await response.json();
console.log(`${book.title} by ${book.author} — ${(book.confidence_score * 100).toFixed(0)}% confidence`);
```

---

## Interactive Documentation

The API is self-documenting via FastAPI's built-in Swagger UI:

- **Swagger UI** (try it in browser): `http://localhost:8000/docs`
- **ReDoc** (clean reference): `http://localhost:8000/redoc`
- **OpenAPI JSON schema**: `http://localhost:8000/openapi.json`

The Swagger UI lets you upload an image directly in the browser and see the
full JSON response without writing any code.
