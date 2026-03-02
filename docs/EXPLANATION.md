# Project Explanation — Book Detection RAG API

A comprehensive study guide for understanding every technology and design decision in this project. Read this top to bottom once, then use it as a reference.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Full Data Flow Walkthrough](#2-full-data-flow-walkthrough)
3. [Why Each Technology Was Chosen](#3-why-each-technology-was-chosen)
4. [Core Concept: What is RAG?](#4-core-concept-what-is-rag)
5. [Core Concept: Vector Embeddings and Semantic Similarity](#5-core-concept-vector-embeddings-and-semantic-similarity)
6. [Core Concept: Cosine Similarity vs Keyword Search](#6-core-concept-cosine-similarity-vs-keyword-search)
7. [Core Concept: HNSW — How ChromaDB Scales](#7-core-concept-hnsw--how-chromadb-scales)
8. [Core Concept: Bi-Encoder vs Cross-Encoder](#8-core-concept-bi-encoder-vs-cross-encoder)
9. [Core Concept: OpenAI Structured Outputs](#9-core-concept-openai-structured-outputs)
10. [Core Concept: LLM Observability with Langfuse](#10-core-concept-llm-observability-with-langfuse)
11. [Core Concept: RAGAS Evaluation Metrics](#11-core-concept-ragas-evaluation-metrics)
12. [Why GPT-4o is Called Twice](#12-why-gpt-4o-is-called-twice)
13. [Confidence Score Design](#13-confidence-score-design)
14. [Architecture Decisions Worth Knowing](#14-architecture-decisions-worth-knowing)
15. [Glossary](#15-glossary)

---

## 1. What This Project Does

You photograph a book — front cover, back cover, or just the spine. You POST that image to this API. Within a few seconds you receive a structured JSON object containing the book's title, author, ISBN, publisher, year, genre, synopsis, and a confidence score.

The pipeline has three stages:

```
Photo of book cover
        │
        ▼  Stage 1 — Vision Extraction
        GPT-4o reads the image like a human would.
        It extracts every visible piece of text:
        title, author, ISBN barcode, subtitle, taglines.
        Output → VisionExtraction (structured Pydantic object)
        │
        ▼  Stage 2 — RAG Retrieval
        The extracted text is converted to a vector (a list of 384 numbers).
        ChromaDB finds the 10 most similar books in its database.
        A cross-encoder re-ranks those 10 → top 3 most relevant candidates.
        Output → list of 3 candidate book records with metadata
        │
        ▼  Stage 3 — LLM Synthesis
        GPT-4o sees the original image, the extraction, and the 3 candidates.
        It cross-references everything, picks the best match, fills in missing
        fields from the database, and assigns a calibrated confidence score.
        Output → BookInfo JSON (title, author, isbn, genre, synopsis, ...)
        │
        ▼  HTTP 200 Response
        DetectionResponse JSON sent back to the caller
```

The key insight: **AI alone cannot reliably identify a book**. GPT-4o may misread text, hallucinate authors, or confuse similar covers. The database acts as a fact-check. The LLM's job in Stage 3 is to *reconcile* what it sees with what the database says — not to invent metadata from scratch.

---

## 2. Full Data Flow Walkthrough

Here is one complete request, traced step by step.

**Request:** POST /api/v1/detect-book with an image of the front cover of *Dune* by Frank Herbert.

### Step 1 — HTTP Validation (detection.py)

FastAPI receives the multipart form upload. The route validates:
- Content-Type is JPEG/PNG/WEBP/GIF (else 415)
- File size ≤ MAX_IMAGE_SIZE_MB (else 413)
- Image bytes can be decoded by Pillow (else 422)

The image is resized if larger than 2048px (to keep OpenAI token costs reasonable) and encoded to base64. Base64 is needed because OpenAI's API accepts images as data URIs (a text protocol), not raw binary.

### Step 2 — Vision Extraction (vision_service.py)

The base64 image is sent to GPT-4o with the instruction: *"Extract all book cover information and return as JSON."*

GPT-4o reads the cover with `"detail": "high"` — this means OpenAI splits the image into 512-pixel tiles and processes each one. This is important for reading small text like ISBN barcodes.

GPT-4o returns JSON like:
```json
{
  "visible_title": "DUNE",
  "visible_author": "Frank Herbert",
  "visible_isbn": "9780441013593",
  "other_text": "40th Anniversary Edition",
  "cover_description": "Orange desert landscape, silhouetted figure"
}
```

This is parsed into a `VisionExtraction` Pydantic model. Fields not visible on the cover become `null`.

### Step 3 — RAG Retrieval (rag_service.py)

All non-null fields are concatenated into a query string:
```
"DUNE Frank Herbert 9780441013593 40th Anniversary Edition Orange desert landscape"
```

The `SentenceTransformer("all-MiniLM-L6-v2")` model converts this text to a 384-dimensional vector. This is an embedding — a point in high-dimensional space where semantically similar texts cluster together.

ChromaDB performs an HNSW approximate nearest-neighbour search and returns the 10 closest book vectors in the database (cosine distance). Each result includes the book's stored text and all its metadata (title, author, ISBN, publisher, etc.).

Then the cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-evaluates all 10 candidates by reading the query and each document together simultaneously. It assigns a precise relevance score and the top 3 are kept.

### Step 4 — Context Formatting (rag_service.py)

The 3 candidates are formatted into a human-readable string like:

```
Candidate books from database (cross-encoder re-ranked) (ranked by relevance):

[1] Title: Dune | Author: Frank Herbert | ISBN: 9780441013593 | Publisher: Ace | ...
    Synopsis: A noble family becomes embroiled in a war for control of the galaxy's...

[2] Title: Dune Messiah | Author: Frank Herbert | ISBN: 9780441172696 | ...
    Synopsis: The second book in the Dune saga...
```

### Step 5 — LLM Synthesis (llm_service.py)

GPT-4o receives a message containing:
1. The original image (for visual confirmation — "does what I see match candidate [1]?")
2. The VisionExtraction JSON
3. The 3 RAG candidates with their metadata

GPT-4o uses OpenAI **Structured Outputs** (`response_format=BookInfo`). This forces the model to produce output that exactly matches the `BookInfo` Pydantic schema — no extra fields, no missing required fields, guaranteed valid JSON.

GPT-4o decides: *"The ISBN extracted from the image exactly matches candidate [1]. Confidence ≥ 0.90."* It fills in `publisher`, `publication_year`, `synopsis`, `tags`, and `rating` from the database record, and returns the complete `BookInfo` object.

### Step 6 — Response

The route assembles the `DetectionResponse`:
```json
{
  "success": true,
  "book": {
    "title": "Dune",
    "author": "Frank Herbert",
    "isbn": "9780441013593",
    "publisher": "Ace Books",
    "publication_year": 1965,
    "genre": "Science Fiction",
    "tags": ["space opera", "political intrigue", "ecology"],
    "synopsis": "A noble family becomes embroiled in an interstellar war...",
    "rating": 4.8,
    "confidence_score": 0.95
  },
  "extraction_notes": "Vision read: 'DUNE' | Author: 'Frank Herbert' | RAG re-ranked 10 → 3 candidate(s)"
}
```

---

## 3. Why Each Technology Was Chosen

### FastAPI (not Flask or Django)

FastAPI is the modern standard for Python APIs in 2024–2025. Three specific reasons it was chosen here:

1. **Pydantic v2 integration**: FastAPI is built on Pydantic. Our `BookInfo` schema is one class — FastAPI uses it for response validation, OpenAPI documentation, and the LLM's Structured Outputs response_format simultaneously. In Flask you would write this three times in three different ways.

2. **async/await**: Reading an uploaded file (`await file.read()`) and calling OpenAI (`await client.chat...`) are I/O operations. FastAPI's async support means the server can handle other requests while waiting. Flask is synchronous by default — one slow OpenAI call blocks the whole worker.

3. **Automatic OpenAPI docs**: The `/docs` endpoint is generated automatically from the Pydantic schemas. No extra configuration needed.

### ChromaDB (not PostgreSQL + pgvector, not Pinecone)

ChromaDB was chosen for its developer experience and zero infrastructure overhead. Comparison:

| Option | Tradeoff |
|--------|----------|
| **ChromaDB** | Embedded — runs in-process, stores to disk. Zero infrastructure. Good for ≤1M vectors. |
| PostgreSQL + pgvector | Requires a running Postgres instance. Better for mixed SQL + vector queries. More ops overhead. |
| Pinecone / Weaviate | Managed cloud service — great for production scale (100M+ vectors). Requires API key + billing. |

For a ~100-book database used in a learning project, ChromaDB is the right call. It demonstrates vector DB concepts without the operational complexity of a managed service.

### sentence-transformers (not OpenAI embeddings)

OpenAI offers `text-embedding-3-small` — so why use a local sentence-transformer?

1. **Cost**: Embedding 100 books + every query at $0.02/1M tokens is negligible. But the sentence-transformer is *free* — zero cost per embedding, which matters if the API serves high traffic or you're seeding a large database.

2. **Latency**: The local model runs in-process (~5ms per embedding). An OpenAI embedding API call adds ~200ms network latency to every request.

3. **Offline capability**: The local model works without internet access (after first download). Important for Docker deployments and testing.

The tradeoff: `all-MiniLM-L6-v2` is a 384-dimension model — smaller and less semantically rich than OpenAI's 3072-dimension `text-embedding-3-large`. For book identification (short, structured text), the smaller model is more than sufficient.

### GPT-4o (not GPT-4o-mini or other vision models)

GPT-4o was chosen for two specific capabilities:

1. **Multimodal vision**: GPT-4o can read text from images with high accuracy, including small print and ISBN barcodes with `"detail": "high"`. GPT-4o-mini has weaker OCR for small text.

2. **Structured Outputs**: Only GPT-4o (and GPT-4o-mini with `beta.parse`) supports the Structured Outputs constrained decoding. This is non-negotiable — we need guaranteed schema-valid JSON.

For cost optimisation in production, `gpt-4o-mini` can be substituted by changing `OPENAI_MODEL=gpt-4o-mini` in `.env`. Accuracy will decrease slightly, especially for degraded images.

### Langfuse (not custom logging)

LLM debugging is fundamentally different from normal API debugging. You cannot grep logs to understand *why* GPT-4o returned a wrong answer. You need to see:
- The exact prompt sent (system + user)
- The exact response received
- Token counts and cost
- Latency per stage
- The trace hierarchy (which stage took longest)

Langfuse provides all of this with two lines of code (`from langfuse.openai import OpenAI` + `@observe`). The alternative is building custom logging middleware — hundreds of lines that you'd inevitably get wrong.

---

## 4. Core Concept: What is RAG?

**RAG = Retrieval-Augmented Generation**

An LLM has a knowledge cutoff date and cannot look things up in real time. If you ask GPT-4o "what is the ISBN of Dune?", it might know, it might not, it might hallucinate a plausible-sounding wrong number.

RAG solves this by injecting relevant *retrieved* information into the LLM's context window before asking it to answer:

```
Without RAG:
  Question → LLM → Answer (from training data, may be wrong)

With RAG:
  Question → Retrieval → Relevant documents
                ↓
  Question + Documents → LLM → Answer (grounded in real data)
```

In this project:
- The "question" is the book cover text extracted by GPT-4o
- The "relevant documents" are the top-3 book records from ChromaDB
- The "answer" is the BookInfo JSON

RAG is the dominant architecture for production LLM applications in 2024–2025. It appears in every serious LLM product: ChatGPT (with Bing), Perplexity, enterprise Q&A systems, code assistants. Understanding it is essential for working in the AI space.

---

## 5. Core Concept: Vector Embeddings and Semantic Similarity

A **vector embedding** is a function that maps text (or images) to a point in high-dimensional space such that *semantically similar texts end up close together*.

```
"Harry Potter and the Sorcerer's Stone" → [0.12, -0.45, 0.89, ...] (384 numbers)
"Harry Potter and the Chamber of Secrets" → [0.13, -0.44, 0.91, ...] (very close)
"Python Programming Tutorial" → [0.91, 0.23, -0.12, ...] (very far)
```

The embedding model (`all-MiniLM-L6-v2`) was trained on hundreds of millions of text pairs labelled as "similar" or "different". Through training, it learned to represent meaning as geometry — similar meaning = nearby points.

**Why this is powerful for book detection:**

If the cover shows "HARRY POTTER" and the database contains "Harry Potter and the Philosopher's Stone", a keyword search fails ("HARRY POTTER" ≠ "Harry Potter and the Philosopher's Stone"). But the embedding similarity is very high — both texts talk about the same entity.

**The 384-number vector:** Each of the 384 dimensions encodes some aspect of meaning. Dimension 47 might correlate with "fictional narrative", dimension 203 with "fantasy genre", etc. The model learned these dimensions automatically — they are not human-assigned.

---

## 6. Core Concept: Cosine Similarity vs Keyword Search

**Keyword search** (like a SQL `LIKE '%dune%'`) matches literal characters. It fails when:
- The cover shows "DUNE" but the DB has "Dune" (case folded — actually fine, but)
- The cover shows "Frank Herbert's classic" and the DB has "Frank Herbert"
- There are typos or OCR errors

**Cosine similarity** measures the angle between two vectors. Two vectors pointing in the same direction have cosine similarity = 1 (identical meaning), pointing opposite have = -1 (opposite meaning), perpendicular = 0 (unrelated).

```
cosine_similarity(a, b) = (a · b) / (|a| × |b|)
```

For L2-normalised vectors (which `all-MiniLM-L6-v2` produces), the formula simplifies to just the dot product `a · b`. This is why ChromaDB is configured with `hnsw:space=cosine` — it is the correct distance metric for normalised vectors.

**ChromaDB stores distances, not similarities.** A distance of 0 = identical, distance of 2 = completely opposite. In the code, similarity is computed as `1 - distance` when displaying results.

**Why not L2 (Euclidean) distance?** For normalised vectors, L2 distance and cosine distance are mathematically equivalent (they produce the same ranking). But if you accidentally use an un-normalised model with cosine space, or a normalised model with L2 space, rankings become incorrect. The rule: match the metric to the model. MiniLM normalises → use cosine.

---

## 7. Core Concept: HNSW — How ChromaDB Scales

If ChromaDB had to compare a query vector against every stored vector, a database of N books would require O(N) comparisons per query. With 100,000 books, that's 100,000 dot products — slow.

**HNSW (Hierarchical Navigable Small World)** is an approximate nearest-neighbour index. It builds a multi-layer graph where each node is a stored vector and edges connect nearby vectors. A search starts at the top layer (long-range connections) and greedily navigates toward the query, zooming in through progressively finer layers.

```
Layer 2 (coarse):   A ─────────── Z
                          ↑
Layer 1 (medium):   A ── M ─── Z
                         ↑
Layer 0 (fine):    A─B─C─M─N─O─Z
                         ↑ query lands here
```

Search complexity: **O(log N)** instead of O(N). For 1 million books, HNSW finds the top-10 nearest neighbours in ~5ms. Linear scan would take seconds.

The tradeoff: HNSW is *approximate* — it might miss the very closest vector occasionally (recall ~99%). For book detection, this is perfectly acceptable — an off-by-one in the top-10 is unimportant when the cross-encoder re-ranks afterwards.

---

## 8. Core Concept: Bi-Encoder vs Cross-Encoder

This is the heart of the two-stage retrieval system.

### Bi-Encoder (Stage A — fast, approximate)

A bi-encoder processes the query and each document **independently**:
```
query → encoder → query_vector
doc_1 → encoder → doc_1_vector   (pre-computed, stored in ChromaDB)
doc_2 → encoder → doc_2_vector
...

similarity = cosine(query_vector, doc_1_vector)
```

The key insight: **document vectors are pre-computed**. You run the embedding model once when seeding the database. At query time, you only embed the query (~5ms), then do dot products against all stored vectors. With HNSW, the search is O(log N).

The limitation: the bi-encoder never sees the query and document *together*. It independently represents each in a shared vector space, but loses nuanced relationships. Two documents might have the same vector even if one is more relevant to a specific query.

### Cross-Encoder (Stage B — slow, precise)

A cross-encoder processes the query and document **together** as a single input:
```
[query + doc_1] → cross-encoder → relevance_score_1
[query + doc_2] → cross-encoder → relevance_score_2
```

Because it sees both texts simultaneously, it can evaluate fine-grained relationships: "does doc_1's author name match the author in the query?", "does doc_1's plot description match the genre cues in the query?". This produces much more accurate rankings.

The limitation: **you cannot pre-compute anything**. Every query requires running the cross-encoder N times. At O(N) and ~100ms per pair, scanning 10,000 books would take 17 minutes.

### The Two-Stage Solution

```
ChromaDB (bi-encoder) fetches top 10 candidates  →  ~5ms
Cross-encoder scores all 10 pairs               →  ~200ms
Top 3 by cross-encoder score → LLM             →  ~2s
```

Total retrieval: ~205ms. Without re-ranking: ~5ms but lower precision. This pattern (bi-encoder coarse filter → cross-encoder precise rerank) is used in every production search system: Google's search pipeline, Cohere Rerank, Pinecone rerank, Elasticsearch's LTR (learning to rank).

---

## 9. Core Concept: OpenAI Structured Outputs

The problem with asking an LLM to "respond in JSON" in a plain prompt:
- It might return markdown-fenced JSON (```json ... ```)
- It might add explanatory text before or after the JSON
- It might return a field with the wrong type (string instead of integer)
- It might omit required fields
- You need a try/except + JSON parser + schema validator + error handling

**Structured Outputs** (OpenAI's `beta.chat.completions.parse`) solves this completely:

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=BookInfo,  # Pass the Pydantic class directly
)
book_info = response.choices[0].message.parsed  # Already a BookInfo instance
```

Internally, OpenAI converts the `BookInfo` Pydantic class into a JSON Schema, then uses **constrained decoding** — a technique where the model's token sampler is filtered to only produce tokens that form valid JSON matching the schema. The model cannot produce invalid output even if it "wants to".

What you get:
- `book_info` is already a fully validated `BookInfo` Python object
- No JSON parsing code
- No validation code
- No error handling for schema mismatches
- `confidence_score` (a required field) is always present

This is one of the most important patterns in production LLM engineering. It converts an unreliable text generator into a reliable structured data extractor.

---

## 10. Core Concept: LLM Observability with Langfuse

When an LLM application returns a wrong answer, how do you debug it?

In a normal API, you grep logs for the request ID and see the inputs/outputs. In an LLM application, you need:
- The exact system prompt and user message sent (prompts change frequently)
- The exact response received (not just the parsed result)
- How many tokens were used (cost tracking)
- How long each stage took (latency breakdown)
- Which stage caused the error

Langfuse provides a **trace dashboard** where every request appears as a tree:

```
detect-book (trace, 3.2s total)
├── vision-extraction (span, 1.1s, 847 input tokens, 89 output tokens)
├── rag-retrieval (span, 0.2s, candidates_fetched=10, candidates_returned=3)
└── llm-synthesis (span, 1.9s, 1203 input tokens, 156 output tokens)
```

**How it works in code:**

```python
# Step 1: Replace the OpenAI import (one line change)
from langfuse.openai import OpenAI  # was: from openai import OpenAI

# Step 2: Decorate each stage
@observe(name="vision-extraction")
def extract_book_text(self, ...):
    ...  # all OpenAI calls inside are automatically traced

@observe(name="rag-retrieval")
def retrieve(self, ...):
    langfuse_context.update_current_observation(
        metadata={"candidates_fetched": 10, "top_similarity": 0.92}
    )
```

The `@observe` decorators form a call stack. The outermost `@observe` (on the route function) creates the parent trace. Each nested `@observe` creates a child span. Langfuse stitches them together using Python's context variable mechanism — no trace ID passing needed.

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are absent from `.env`, the `@observe` decorator becomes a no-op. Zero performance impact when observability is disabled.

---

## 11. Core Concept: RAGAS Evaluation Metrics

How do you know if your RAG pipeline is good? RAGAS provides four metrics, each measuring a different failure mode.

### Faithfulness — "Did the LLM stay honest?"

Measures whether every claim in the LLM's answer is supported by the retrieved context.

```
Retrieved context: "Dune was published in 1965 by Chilton Books"
LLM answer:       "Dune was published in 1965 by Ace Books"

Faithfulness = LOW — "Ace Books" is not in the context (LLM hallucinated the publisher)
```

Low faithfulness = the LLM is inventing facts not present in the retrieved documents. Fix: tighten the system prompt ("only use information from the provided database candidates").

### Context Precision — "Was retrieved context relevant?"

Measures what proportion of retrieved chunks were actually useful for answering the question.

```
Retrieved: [Dune record, Dune Messiah record, Foundation record]
Useful for answering: only Dune record

Context Precision = 1/3 ≈ 0.33 (low — two out of three retrieved docs were noise)
```

Low context precision = ChromaDB is returning irrelevant books. Fix: improve the query string construction; tune `RAG_FETCH_K`; improve the bi-encoder or switch to a domain-specific model.

### Context Recall — "Did we retrieve all needed information?"

Measures whether all ground-truth information can be found in the retrieved context.

```
Ground truth: "Dune by Frank Herbert, ISBN 9780441013593"
Retrieved context: contains Dune record with ISBN and author

Context Recall = HIGH — all ground truth info is present in context
```

Low context recall = the correct book is not in ChromaDB (or was not retrieved). Fix: add missing books to the database; increase `RAG_FETCH_K`.

### Answer Relevancy — "Did the LLM actually answer the question?"

Measures whether the LLM's answer is relevant to the question asked. A verbose answer that goes off-topic scores low.

```
Question: "What book is shown in this cover image?"
Answer:   "The book shown appears to be Dune by Frank Herbert, published in 1965."
Answer Relevancy = HIGH

Answer: "Books are wonderful objects that have shaped human civilization..."
Answer Relevancy = VERY LOW
```

Low answer relevancy usually indicates a prompt engineering issue.

---

## 12. Why GPT-4o is Called Twice

This is the most common question about the pipeline. Why not do it all in one call?

**Stage 1 (vision_service.py):** *Extraction only.* GPT-4o is given one job: read the cover and extract the text. The system prompt says *"only report what is literally visible — do not invent or guess."* The model acts as a precise OCR + structured text extractor.

**Stage 3 (llm_service.py):** *Synthesis + cross-referencing.* GPT-4o is given a harder job: look at the image AND the database candidates AND the extracted text, then decide which candidate is the correct book, fill in missing metadata from the database, and assign a calibrated confidence score.

**Why separate these?**

If you tried to do both in one call:
- The extraction instructions would conflict with the synthesis instructions
- You cannot ask an LLM to "only report visible text" and "fill in metadata from database" simultaneously
- The RAG candidates are not available at extraction time — you need the extraction *first* to know what to search for

**Why does Stage 3 see the image again?**

The LLM needs to *visually confirm* the database candidate. Consider this scenario:
- The cover shows what looks like "Dune" but is actually "June" (similar typography)
- The RAG step retrieves "Dune" as the top candidate
- If Stage 3 only sees the text extraction ("DUNE"), it will confidently return Dune

But if Stage 3 sees the original image alongside the extraction, it can notice: "wait, the cover art looks like a garden scene (June) not a desert (Dune)". Visual confirmation prevents this class of error.

---

## 13. Confidence Score Design

The confidence score is not a raw model probability — it is a deterministic rule applied by GPT-4o based on the evidence.

| Score Range | Condition | Meaning |
|-------------|-----------|---------|
| ≥ 0.95 | ISBN visible in image AND matches a database record exactly | Highest certainty — ISBNs are globally unique identifiers |
| 0.90–0.94 | Title AND author clearly readable, database candidate matches with good similarity (≥ 0.60) | Very high certainty — two signals agree plus DB confirmation |
| 0.80–0.89 | Title AND author clearly readable, database match with lower similarity | High certainty — text evidence strong, DB match weaker |
| 0.70–0.79 | Title AND author clearly readable, even without a database match | Good certainty — text evidence alone is strong |
| 0.40–0.69 | Only title OR author readable (partial signal) | Medium certainty — partial match, could be different edition |
| < 0.40 | No text evidence; visual inference alone | Low certainty — LLM inference from image only |

**Why these thresholds?**

ISBN is a globally unique identifier — if the barcode matches, there is almost no possibility of error (≥0.95, not 1.0, because OCR errors can corrupt ISBNs).

Title + Author agreement means two independent signals point to the same book. With a good database similarity score (≥ 0.60), this warrants ≥ 0.90 confidence. Without database confirmation, text evidence alone still justifies 0.70–0.79.

**ISBN population from database:** When confidence is ≥ 0.90 but no ISBN is visible in the image, the ISBN is populated from the matched database record — it is reliable enough to include.

Title-only matches are weaker — many books share common words in titles ("The Great War", "Gone Girl" if the author is misread).

Below 0.40 means we are basically guessing from the image alone. This score makes the uncertainty transparent to the caller, who can decide whether to accept or reject the identification.

---

## 14. Architecture Decisions Worth Knowing

### Lifespan Resource Management

The sentence-transformer model (`all-MiniLM-L6-v2`) is ~80MB on disk and takes ~2 seconds to load into RAM. The CrossEncoder is another ~80MB and ~1 second to load.

If you loaded these models on every request, a simple image upload would take 3+ seconds just for model loading before any computation begins. The solution is to load once at startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE when the server starts
    vector_store = VectorStore(settings)
    vector_store.initialize()  # Loads ~80MB model into RAM
    app.state.vector_store = vector_store

    reranker = CrossEncoder(settings.RERANKER_MODEL)  # Another ~80MB
    app.state.reranker = reranker
    yield  # Server runs here
    # Cleanup on shutdown
```

`app.state` is FastAPI's built-in mechanism for storing application-level state. The `get_vector_store()` and `get_reranker()` dependency functions simply read from `app.state` — they are O(1) attribute lookups.

### Why the Embedding Model Must Match the Database

All book vectors in ChromaDB were generated by `all-MiniLM-L6-v2`. When a query arrives, it is also embedded by `all-MiniLM-L6-v2`. The cosine similarity is meaningful because both vectors live in the same geometric space.

If you change `EMBEDDING_MODEL` to a different model and query with a new vector, you are comparing apples and oranges — vectors from two different spaces. The similarity scores will be meaningless. You would see high confidence returns for completely wrong books.

This is why the CLAUDE.md lists this as an architecture invariant: **changing `EMBEDDING_MODEL` requires re-running `populate_db.py`** to regenerate all stored vectors with the new model.

### Why Cosine Not L2

`all-MiniLM-L6-v2` produces **L2-normalised** vectors (all vectors have length exactly 1.0). For unit-length vectors:

```
cosine_distance(a, b) = 1 - (a · b)
L2_distance(a, b)²    = 2 - 2(a · b)
```

These are mathematically equivalent for unit vectors — they produce identical rankings. However, ChromaDB still needs to know which metric you intend to use, because it stores the metric metadata and uses it for HNSW index construction. Setting `hnsw:space=cosine` explicitly documents the intent and ensures correctness if the model ever changes.

### Pydantic Settings with @lru_cache

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`Settings()` reads all environment variables and validates them on construction. The `@lru_cache` decorator makes `get_settings()` return the same `Settings` instance on every call (memoisation). This ensures:
- Environment variables are read once (no repeated disk/env access)
- All parts of the app share the same config object
- Tests can override `get_settings` in FastAPI's dependency injection to inject test settings

---

## 15. Glossary

**Base64**: An encoding that converts binary data (image bytes) to ASCII text. Used because HTTP APIs (and OpenAI's image API) communicate in text, not raw binary. A base64 image is ~33% larger than the original binary.

**Bi-encoder**: A model that independently encodes a query and a document into vectors. Fast because document vectors are pre-computed. Less accurate than a cross-encoder because it never sees query+document together.

**ChromaDB**: An open-source vector database. Stores vectors alongside metadata and supports similarity search. Uses HNSW for fast approximate nearest-neighbour lookup.

**Confidence score**: A number from 0 to 1 representing how certain the model is about its book identification. Computed by deterministic rules in the LLM system prompt, not a raw probability.

**Constrained decoding**: A technique where the LLM's token sampling is filtered to only allow tokens that produce output matching a schema. OpenAI Structured Outputs uses this to guarantee valid JSON.

**Cosine similarity**: A measure of similarity between two vectors, computed as the cosine of the angle between them. 1 = identical direction, 0 = perpendicular (unrelated), -1 = opposite. Ranges from 1 (most similar) to -1 (most different).

**Cross-encoder**: A model that processes a query and document together as a single input, outputting a relevance score. Highly accurate but cannot pre-compute — O(N) inference time per query.

**Data URI**: A URI scheme that embeds binary data directly in the URI string. Format: `data:<mediatype>;base64,<data>`. Used to send images to OpenAI's API without a file upload endpoint.

**Dependency injection (FastAPI Depends)**: A pattern where functions declare their dependencies as function parameters. FastAPI resolves and caches these automatically. Used here to inject `VectorStore` and `CrossEncoder` from `app.state` into route handlers.

**Embedding**: A vector representation of text (or other data) in a high-dimensional space. Embeddings are the output of an embedding model. Semantically similar texts have similar (nearby) embeddings.

**Faithfulness (RAGAS)**: A metric measuring whether the LLM's answer is grounded in the retrieved context. Low faithfulness = hallucination.

**HNSW (Hierarchical Navigable Small World)**: A graph-based approximate nearest-neighbour index. Enables O(log N) similarity search instead of O(N) linear scan. Used internally by ChromaDB.

**L2-normalised**: A vector where the Euclidean length (L2 norm) is exactly 1.0. SentenceTransformer models output L2-normalised vectors. For such vectors, cosine similarity and dot product are equivalent.

**Langfuse**: An open-source LLM observability platform. Captures prompts, responses, token counts, latency, and trace hierarchies for debugging and monitoring LLM applications.

**Lifespan (FastAPI)**: An `asynccontextmanager` function that runs setup code before the server accepts requests and teardown code after it shuts down. Used here to load the embedding model and cross-encoder once at startup.

**LLM (Large Language Model)**: A neural network trained on vast amounts of text to predict and generate human language. GPT-4o is an LLM. In this project, it acts as both an OCR engine (Stage 1) and a reasoning/synthesis engine (Stage 3).

**Multimodal**: A model that processes multiple types of input (text + images + audio). GPT-4o is multimodal — it can read text from images and reason about visual content.

**Pydantic**: A Python library for data validation and settings management. Models are defined as Python classes with type annotations. Pydantic v2 validates data at construction time and integrates natively with FastAPI.

**RAG (Retrieval-Augmented Generation)**: An architecture pattern where relevant documents are retrieved from a database and injected into the LLM's context before asking it to generate a response. Grounds the LLM in real data and reduces hallucination.

**RAGAS**: An evaluation framework for RAG pipelines. Computes four metrics (faithfulness, context precision, context recall, answer relevancy) using an LLM as a judge.

**Reranking**: The process of re-ordering an initial set of retrieved results using a more accurate (but slower) model. In this project, 10 bi-encoder candidates are re-ordered by a cross-encoder to surface the most relevant top-3.

**Sentence-transformers**: A Python library providing pre-trained embedding models (bi-encoders) and cross-encoders optimised for semantic similarity tasks. `all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` are both sentence-transformer models.

**Structured Outputs**: An OpenAI feature (`beta.chat.completions.parse`) that uses constrained decoding to guarantee the LLM's response matches a provided JSON schema. The response is automatically parsed into a Pydantic model.

**Vector**: A list of numbers representing a point in high-dimensional space. In this project, text is converted to 384-dimensional vectors by the embedding model.

**Vector database**: A database optimised for storing and searching vectors by similarity (nearest-neighbour search). ChromaDB is a vector database. Traditional SQL databases are not designed for this operation.

**VisionExtraction**: The Pydantic model capturing everything GPT-4o reads from the book cover image: `visible_title`, `visible_author`, `visible_isbn`, `other_text`, `cover_description`. This is the output of Stage 1 and the input to Stage 2.
