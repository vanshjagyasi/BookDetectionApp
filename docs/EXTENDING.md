# Extending the Project

Step-by-step guides for common extension tasks.

---

## Table of Contents

1. [Add a new field to BookInfo](#1-add-a-new-field-to-bookinfo)
2. [Switch to Anthropic Claude](#2-switch-to-anthropic-claude)
3. [Switch to Google Gemini](#3-switch-to-google-gemini)
4. [Use a local model via Ollama](#4-use-a-local-model-via-ollama)
5. [Change the embedding model](#5-change-the-embedding-model)
6. [Add a new API endpoint](#6-add-a-new-api-endpoint)
7. [Scale with a managed vector database](#7-scale-with-a-managed-vector-database)
8. [Add more books to the database](#8-add-more-books-to-the-database)

---

## 1. Add a New Field to BookInfo

Example: add a `language` field.

**Step 1** — Add to `app/schemas/book.py`:
```python
class BookInfo(BaseModel):
    ...
    language: Optional[str] = Field(None, description="Language of the book (e.g. 'English').")
```

**Step 2** — Tell the LLM about it in `app/services/llm_service.py`:
```python
LLM_SYSTEM_PROMPT = """...
- language: the language the book is written in (e.g. 'English', 'Spanish')
...
"""
```
OpenAI Structured Outputs will automatically enforce the new field from the
updated `BookInfo` schema — no other changes needed in `llm_service.py`.

**Step 3** — (Optional) If the field should come from the database rather than
being inferred, add it to the metadata dict in `scripts/populate_db.py`:
```python
metadata = {
    ...
    "language": info.get("language") or "en",
}
```
Then update `rag_service.py` `format_context()` to include it in the context string.

**Step 4** — Re-run `python scripts/populate_db.py` if you added it to metadata.

---

## 2. Switch to Anthropic Claude

Replace OpenAI GPT-4o with Anthropic Claude claude-sonnet-4-6 (multimodal + tool_use).

**Files to change:** `vision_service.py`, `llm_service.py`, `config.py`, `requirements.txt`

### `requirements.txt`
```diff
-openai==1.54.0
+anthropic==0.34.2
```

### `app/config.py`
```diff
-OPENAI_API_KEY: str
-OPENAI_MODEL: str = "gpt-4o"
+ANTHROPIC_API_KEY: str
+ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
```

### `app/services/vision_service.py`
```python
import anthropic
import json
from app.config import Settings
from app.schemas.book import VisionExtraction

class VisionService:
    def __init__(self, settings: Settings) -> None:
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.ANTHROPIC_MODEL

    def extract_book_text(self, image_b64: str, media_type: str) -> VisionExtraction:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=VISION_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": "Extract all book information as JSON."}
                ],
            }],
        )
        raw = message.content[0].text
        if "```" in raw:
            raw = raw.split("```json")[-1].split("```")[0].strip()
        return VisionExtraction.model_validate(json.loads(raw))
```

### `app/services/llm_service.py`
Anthropic uses `tool_use` instead of Structured Outputs:
```python
import anthropic
import json
from app.config import Settings
from app.schemas.book import BookInfo, VisionExtraction

BOOK_INFO_TOOL = {
    "name": "return_book_info",
    "description": "Return structured book identification results",
    "input_schema": BookInfo.model_json_schema(),
}

class LLMService:
    def __init__(self, settings: Settings) -> None:
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.ANTHROPIC_MODEL

    def generate_book_info(self, extraction, rag_context, image_b64, media_type) -> BookInfo:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=LLM_SYSTEM_PROMPT,
            tools=[BOOK_INFO_TOOL],
            tool_choice={"type": "tool", "name": "return_book_info"},
            messages=[{"role": "user", "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                },
                {
                    "type": "text",
                    "text": f"## Vision\n{extraction.model_dump_json()}\n\n## {rag_context}"
                }
            ]}],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return BookInfo(**tool_block.input)
```

### `.env`
```diff
-OPENAI_API_KEY=sk-...
-OPENAI_MODEL=gpt-4o
+ANTHROPIC_API_KEY=sk-ant-...
+ANTHROPIC_MODEL=claude-sonnet-4-6
```

---

## 3. Switch to Google Gemini

Google's Gemini 1.5 Pro supports vision and structured output.

**Key differences from OpenAI:**
- Uses `google-generativeai` SDK
- Image format: `types.Part.from_data(data=bytes, mime_type=media_type)`
- Structured output: `generation_config={"response_schema": schema, "response_mime_type": "application/json"}`

Install: `pip install google-generativeai`

The pattern is very similar to Claude's approach. See the Gemini Python SDK docs
for the exact multimodal message format.

---

## 4. Use a Local Model via Ollama

Ollama exposes an OpenAI-compatible API endpoint, so you only need to change
the `base_url` and `model` — **no code changes required**.

```bash
# Install and run Ollama
ollama serve
ollama pull llava   # multimodal model
```

In `.env`:
```env
OPENAI_API_KEY=ollama      # Ollama doesn't need a real key
OPENAI_MODEL=llava
```

In `app/services/vision_service.py` and `app/services/llm_service.py`,
change the OpenAI client initialisation:
```python
self.client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)
```

**Note:** Local models do not support OpenAI Structured Outputs (`beta.parse`).
Switch `llm_service.py` to use `response_format={"type": "json_object"}` and
manually parse + validate with `BookInfo.model_validate(json.loads(content))`.

---

## 5. Change the Embedding Model

To use a different sentence-transformers model (e.g., `all-mpnet-base-v2`
for higher quality, or `all-MiniLM-L12-v2` for a balance):

1. Update `.env`:
   ```env
   EMBEDDING_MODEL=all-mpnet-base-v2
   ```

2. **Delete the existing ChromaDB data** (vectors from the old model are incompatible):
   ```bash
   rm -rf data/chroma_db
   ```

3. Re-seed the database:
   ```bash
   python scripts/populate_db.py
   ```

**Why you must re-seed:** A 384-dim vector (MiniLM) and a 768-dim vector
(MPNet) live in completely different spaces. Mixing them produces meaningless
similarity scores. Deleting and rebuilding ensures all vectors use the same model.

---

## 6. Add a New API Endpoint

Example: `GET /api/v1/books/search?q=dune` — direct text search without an image.

**Step 1** — Create `app/api/v1/routes/search.py`:
```python
from fastapi import APIRouter, Depends, Query
from app.db.vector_store import VectorStore
from app.dependencies import get_vector_store

router = APIRouter(tags=["Search"])

@router.get("/books/search")
async def search_books(
    q: str = Query(..., description="Search query"),
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    results = vector_store.query(q, n_results=5)
    return {"results": [r["metadata"] for r in results]}
```

**Step 2** — Register in `app/main.py`:
```python
from app.api.v1.routes.search import router as search_router
...
app.include_router(search_router, prefix="/api/v1")
```

**Step 3** — Add tests in `tests/test_search.py`.

---

## 7. Scale with a Managed Vector Database

For production at scale, swap ChromaDB for Qdrant or Pinecone.

### Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
pip install qdrant-client
```

Replace `app/db/vector_store.py` with a Qdrant client.
The interface (`embed()`, `upsert()`, `query()`, `count()`) stays the same —
only the underlying client changes. All other code remains untouched.

### Pinecone
```bash
pip install pinecone-client
```

Same approach — implement the same `VectorStore` interface using the Pinecone SDK.

---

## 8. Add More Books to the Database

### Option A: Add more genre queries
Edit `SEARCH_QUERIES` in `scripts/populate_db.py` and re-run:
```bash
python scripts/populate_db.py
```
Safe to re-run — existing books are upserted (not duplicated).

### Option B: Load from a custom CSV
Add to `scripts/populate_db.py`:
```python
import csv

def load_from_csv(csv_path: str, store: VectorStore) -> int:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            doc = f"{row['title']} by {row['author']}. {row.get('description', '')}"
            store.upsert(
                book_id=row.get('isbn') or hashlib.md5(doc.encode()).hexdigest(),
                document=doc,
                metadata={k: row.get(k, '') for k in
                    ['title','author','isbn','publisher','publication_year',
                     'genre','tags','synopsis','price','rating']},
            )
            count += 1
    return count
```

### Option C: Use Open Library API
Open Library (`https://openlibrary.org/api/books`) is free and has millions of records.
Replace the Google Books API calls in `populate_db.py` with Open Library queries.
No API key required.
