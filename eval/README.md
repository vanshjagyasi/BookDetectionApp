# RAGAS Evaluation Harness

Offline quality evaluation for the Book Detection RAG pipeline using the [RAGAS](https://docs.ragas.io) framework.

---

## Quick Start

```bash
# 1. Ensure ChromaDB is populated
python scripts/populate_db.py

# 2. Run the full evaluation (18 golden entries, ~3–5 min, uses OpenAI API)
python eval/run_eval.py

# 3. Evaluate a subset (faster — good for debugging)
python eval/run_eval.py --limit 5

# 4. Save detailed results to JSON
python eval/run_eval.py --output eval/results.json

# 5. Compare with re-ranking disabled
python eval/run_eval.py --no-rerank
```

**Requirements:** `OPENAI_API_KEY` in `.env`, ChromaDB populated, `pip install ragas datasets`.

---

## What the Evaluation Measures

The harness tests the **RAG retrieval + LLM synthesis** stages (Stages 2 and 3) without real images. Each golden entry supplies a synthetic `VisionExtraction` — the text that GPT-4o *would* read from a real cover — bypassing Stage 1 (vision) entirely.

### The Four RAGAS Metrics

#### 1. Faithfulness
> *Does the LLM answer stay grounded in the retrieved context?*

- **Score 1.0** — every claim in the answer is supported by a retrieved document
- **Score 0.0** — the LLM hallucinated information not present in any retrieved chunk
- **What to do if low:** Tighten `LLM_SYSTEM_PROMPT` to discourage invention; consider increasing `RERANK_TOP_K` so more context reaches the LLM.

#### 2. Context Precision
> *Are the top-ranked retrieved chunks actually relevant to the question?*

- **Score 1.0** — every retrieved document contains useful information for identification
- **Score 0.0** — retrieved documents are mostly irrelevant noise
- **What to do if low:** Tune the RAG query construction in `rag_service.py`; experiment with `EMBEDDING_MODEL`; expand the ChromaDB collection.

#### 3. Context Recall
> *Did retrieval surface all the information needed to produce the correct answer?*

- **Score 1.0** — everything in the ground truth can be found in the retrieved context
- **Score 0.0** — the correct book was not retrieved at all
- **What to do if low:** The book is missing from ChromaDB — run `populate_db.py` to add more titles; or increase `RAG_FETCH_K` so more candidates are fetched before re-ranking.

#### 4. Answer Relevancy
> *Is the LLM's answer actually relevant to the identification question?*

- **Score 1.0** — the answer directly addresses what book was shown
- **Score 0.0** — the answer goes off-topic or returns empty fields
- **What to do if low:** Review `LLM_SYSTEM_PROMPT` in `llm_service.py`; ensure `BookInfo` fields are clearly described.

### Score Interpretation

| Average Score | Interpretation |
|--------------|---------------|
| ≥ 0.80 | Excellent — pipeline is working well |
| 0.65–0.79 | Good — minor tuning may improve results |
| 0.50–0.64 | Fair — expand golden dataset or tune prompts |
| < 0.50 | Needs work — check DB population and prompt quality |

---

## The Golden Dataset

`eval/golden_dataset.json` contains 18 hand-crafted test cases covering all 12 database genres:

| Genre | Examples |
|-------|---------|
| Science Fiction | Dune, Project Hail Mary, The Martian |
| Fiction | 1984, The Great Gatsby, To Kill a Mockingbird |
| Fantasy | The Way of Kings, Harry Potter |
| Mystery/Thriller | Gone Girl |
| Biography/Memoir | Educated |
| History | Sapiens |
| Self-Help | Atomic Habits |
| Computers | Python Crash Course, Hands-On ML |
| Young Adult | The Hunger Games |
| Romance | The Notebook |
| Science | A Brief History of Time |
| Psychology | Thinking, Fast and Slow |

The dataset also includes **degraded-signal test cases** to stress-test the pipeline:

- **`title-only-no-isbn`** — no ISBN visible (tests bi-encoder semantic matching)
- **`partial-title-author-only`** — only partial title + no author (tests LLM inference)
- **`spine-only-minimal-info`** — spine view with no ISBN (tests minimal-signal retrieval)

### Golden Dataset Format

```json
{
  "id": "unique-slug",
  "vision_extraction": {
    "visible_title": "...",
    "visible_author": "...",
    "visible_isbn": "...",
    "other_text": "...",
    "cover_description": "..."
  },
  "ground_truth": {
    "title": "...",
    "author": "...",
    "isbn": "...",
    "genre": "..."
  }
}
```

`vision_extraction` mirrors the `VisionExtraction` Pydantic model — it is the synthetic output of what GPT-4o *would* read from a real cover photo.

`ground_truth` is the authoritative answer used to compute `context_recall`.

### Adding New Golden Entries

1. Pick a book that is (or should be) in the ChromaDB collection.
2. Write a realistic `vision_extraction` — imagine what text would be visible on a real cover photo of that book.
3. Set `ground_truth` with the canonical title, author, ISBN-13, and genre.
4. Add the entry to `eval/golden_dataset.json`.
5. Re-run `python eval/run_eval.py` to see how the new case scores.

**Tips for good golden entries:**
- Include entries where the ISBN *is* visible (high-confidence cases)
- Include entries where only title/author are visible (medium-confidence)
- Include entries with only partial information (stress tests)
- Aim for genre balance — at least one entry per DB genre

---

## How RAGAS Works Internally

RAGAS uses an LLM (by default the same OpenAI model set in your `.env`) as a *judge* to score each metric. It sends structured prompts asking things like:

> "Given the retrieved context and this answer, score how faithful the answer is on a scale of 0–1."

This means:
- **RAGAS costs tokens** — each evaluation entry makes several LLM API calls
- **Results vary slightly** between runs (LLM scoring has variance)
- **The LLM judge is imperfect** — RAGAS scores are estimates, not ground truth

For a more deterministic comparison (e.g. A/B testing re-ranking vs. no re-ranking), run each configuration 3× and average the scores.

---

## Comparing Configurations

```bash
# With re-ranking (default)
python eval/run_eval.py --output eval/results_reranked.json

# Without re-ranking
python eval/run_eval.py --no-rerank --output eval/results_no_rerank.json

# Compare the JSON files to see metric improvements from re-ranking
```

Expected improvement from re-ranking: `context_precision` should increase most noticeably, since the cross-encoder places the most relevant documents at the top of the context window.

---

## Troubleshooting

**`ChromaDB collection is empty`**
→ Run `python scripts/populate_db.py` first.

**`OPENAI_API_KEY not set`**
→ Copy `.env.example` to `.env` and add your key.

**`ModuleNotFoundError: ragas`**
→ Run `pip install ragas datasets`.

**Very low context_recall scores**
→ The books in the golden dataset may not be in ChromaDB. Run `python scripts/verify_db.py` to inspect what titles are present. If missing, the Google Books API seeding may not have included them — add their titles to `SEARCH_QUERIES` in `scripts/populate_db.py` and re-seed.

**RAGAS hangs or times out**
→ Use `--limit 3` to test with a small subset first. RAGAS makes multiple LLM calls per entry, so 18 entries × ~4 API calls = ~72 OpenAI requests.
