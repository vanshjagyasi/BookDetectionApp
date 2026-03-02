"""
eval/run_eval.py
================
Offline RAGAS evaluation harness for the Book Detection pipeline.

What this does:
  1. Loads the golden dataset from eval/golden_dataset.json.
  2. For each entry, runs Stage 2 (RAGService) + Stage 3 (LLMService)
     using the synthetic VisionExtraction — no real images needed.
  3. Collects RAGAS inputs: question, answer, contexts, ground_truth.
  4. Computes four RAGAS metrics and prints a summary report.

Metrics computed:
  - faithfulness:        Does the answer stay faithful to the retrieved context?
  - context_precision:  Are the top retrieved chunks actually relevant?
  - context_recall:     Did retrieval surface all information needed for the answer?
  - answer_relevancy:   Is the LLM answer relevant to the question asked?

Requirements:
  - OPENAI_API_KEY set in .env (used by LLMService for GPT-4o calls)
  - ChromaDB must be populated: run `python scripts/populate_db.py` first
  - Install: pip install ragas datasets

Usage:
    # From the project root:
    python eval/run_eval.py

    # Evaluate a subset (first 5 entries):
    python eval/run_eval.py --limit 5

    # Save results to JSON:
    python eval/run_eval.py --output eval/results.json

    # Disable re-ranking (faster, uses bi-encoder order only):
    python eval/run_eval.py --no-rerank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root: python eval/run_eval.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import get_settings
from app.db.vector_store import VectorStore
from app.schemas.book import VisionExtraction
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

# Placeholder base64 image: 1×1 white PNG (the LLM stage needs an image arg,
# but RAGAS eval uses the text-only path — the image contributes nothing when
# the golden dataset already supplies the ground-truth extraction).
_WHITE_1X1_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)


def load_golden_dataset(path: Path) -> list[dict]:
    """Load and validate the golden dataset JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {path}. "
            "Expected: eval/golden_dataset.json"
        )
    with path.open() as f:
        data = json.load(f)
    print(f"Loaded {len(data)} golden entries from {path}")
    return data


def build_query_string(extraction: VisionExtraction) -> str:
    """Replicate the RAGService query construction for the 'question' field."""
    parts = [
        extraction.visible_title,
        extraction.visible_author,
        extraction.cover_description,
        extraction.other_text,
    ]
    return " ".join(p for p in parts if p and p.strip())


def ground_truth_to_string(gt: dict) -> str:
    """Convert a ground_truth dict to a plain string for RAGAS context_recall."""
    parts = []
    if gt.get("title"):
        parts.append(f"Title: {gt['title']}")
    if gt.get("author"):
        parts.append(f"Author: {gt['author']}")
    if gt.get("isbn"):
        parts.append(f"ISBN: {gt['isbn']}")
    if gt.get("genre"):
        parts.append(f"Genre: {gt['genre']}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    limit: int | None = None,
    no_rerank: bool = False,
    output_path: Path | None = None,
) -> dict:
    """
    Run the RAGAS evaluation pipeline.

    Args:
        limit:       If set, evaluate only the first N golden entries.
        no_rerank:   If True, skip CrossEncoder re-ranking (faster).
        output_path: If set, write the per-entry results JSON here.

    Returns:
        Dict of metric names → float scores.
    """
    settings = get_settings()

    # --- Set up VectorStore ---
    print("\nInitialising VectorStore…")
    vector_store = VectorStore(settings)
    vector_store.initialize()
    print(f"  Collection size: {vector_store.count()} books\n")

    if vector_store.count() == 0:
        print(
            "ERROR: ChromaDB collection is empty.\n"
            "Run:  python scripts/populate_db.py\n"
            "Then re-run the evaluation."
        )
        sys.exit(1)

    # --- Optionally load CrossEncoder ---
    reranker = None
    if not no_rerank and settings.RERANKER_MODEL:
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            print(f"Loading CrossEncoder: {settings.RERANKER_MODEL}")
            reranker = CrossEncoder(settings.RERANKER_MODEL)
            print("  CrossEncoder ready.\n")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: Could not load CrossEncoder ({exc}). Falling back to bi-encoder order.\n")

    # --- Initialise services ---
    rag_svc = RAGService(vector_store, reranker)
    llm_svc = LLMService(settings)

    # --- Load golden dataset ---
    entries = load_golden_dataset(GOLDEN_DATASET_PATH)
    if limit:
        entries = entries[:limit]
        print(f"Evaluating first {limit} entries.\n")

    # --- Collect RAGAS inputs ---
    ragas_questions: list[str] = []
    ragas_answers: list[str] = []
    ragas_contexts: list[list[str]] = []
    ragas_ground_truths: list[str] = []

    per_entry_results = []

    for i, entry in enumerate(entries, 1):
        entry_id = entry["id"]
        print(f"[{i}/{len(entries)}] Evaluating: {entry_id}")

        extraction = VisionExtraction(**entry["vision_extraction"])
        gt = entry["ground_truth"]

        # Stage 2: RAG retrieval
        rag_results = rag_svc.retrieve(extraction)
        rag_context = rag_svc.format_context(rag_results)

        # Stage 3: LLM synthesis (uses placeholder image — text evidence dominates)
        try:
            book_info = llm_svc.generate_book_info(
                extraction,
                rag_context,
                _WHITE_1X1_PNG_B64,
                "image/png",
            )
            answer = (
                f"Title: {book_info.title} | "
                f"Author: {book_info.author} | "
                f"ISBN: {book_info.isbn} | "
                f"Genre: {book_info.genre} | "
                f"Confidence: {book_info.confidence_score}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR generating book info: {exc}")
            answer = f"ERROR: {exc}"

        # Contexts = each retrieved document text (what the LLM actually saw)
        contexts = [r["document"] for r in rag_results] if rag_results else ["No context retrieved."]

        question = build_query_string(extraction)
        ground_truth = ground_truth_to_string(gt)

        ragas_questions.append(question)
        ragas_answers.append(answer)
        ragas_contexts.append(contexts)
        ragas_ground_truths.append(ground_truth)

        per_entry_results.append(
            {
                "id": entry_id,
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "context_count": len(contexts),
                "rag_results": len(rag_results),
            }
        )
        print(f"  Answer: {answer[:100]}…" if len(answer) > 100 else f"  Answer: {answer}")
        print(f"  Contexts retrieved: {len(contexts)}")

    # --- Build RAGAS Dataset ---
    print("\nBuilding RAGAS dataset…")
    ragas_dataset = Dataset.from_dict(
        {
            "question": ragas_questions,
            "answer": ragas_answers,
            "contexts": ragas_contexts,
            "ground_truth": ragas_ground_truths,
        }
    )

    # --- Run RAGAS evaluation ---
    print("Running RAGAS metrics (this makes OpenAI API calls)…\n")
    results = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            context_precision,
            context_recall,
            answer_relevancy,
        ],
    )

    # --- Print report ---
    scores = results.to_pandas().mean(numeric_only=True).to_dict()
    _print_report(scores, len(entries), reranker is not None)

    # --- Save per-entry results if requested ---
    if output_path:
        output_data = {
            "config": {
                "n_entries": len(entries),
                "reranking_enabled": reranker is not None,
                "reranker_model": settings.RERANKER_MODEL if reranker else None,
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_model": settings.OPENAI_MODEL,
                "rag_fetch_k": settings.RAG_FETCH_K,
                "rerank_top_k": settings.RERANK_TOP_K,
            },
            "scores": scores,
            "per_entry": per_entry_results,
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nFull results saved to: {output_path}")

    return scores


def _print_report(scores: dict, n_entries: int, reranked: bool) -> None:
    """Print a formatted summary of RAGAS metric scores."""
    print("=" * 60)
    print("  RAGAS Evaluation Report")
    print("=" * 60)
    print(f"  Entries evaluated : {n_entries}")
    print(f"  Re-ranking        : {'enabled' if reranked else 'disabled (bi-encoder only)'}")
    print("-" * 60)

    metric_labels = {
        "faithfulness":       "Faithfulness      (answer grounded in context)",
        "context_precision":  "Context Precision (relevant chunks ranked high)",
        "context_recall":     "Context Recall    (all needed info retrieved)",
        "answer_relevancy":   "Answer Relevancy  (answer addresses the question)",
    }

    for key, label in metric_labels.items():
        score = scores.get(key, float("nan"))
        bar = _score_bar(score)
        print(f"  {label:<45} {score:.3f}  {bar}")

    print("=" * 60)

    # Interpretation
    avg = sum(v for v in scores.values() if v == v) / max(len(scores), 1)
    if avg >= 0.80:
        verdict = "Excellent — pipeline is working well."
    elif avg >= 0.65:
        verdict = "Good — minor tuning may improve results."
    elif avg >= 0.50:
        verdict = "Fair — consider expanding the golden dataset or tuning prompts."
    else:
        verdict = "Needs work — check DB population and prompt quality."

    print(f"\n  Average score: {avg:.3f}  →  {verdict}\n")


def _score_bar(score: float, width: int = 10) -> str:
    """Return a simple ASCII bar for a 0–1 score."""
    if score != score:  # NaN check
        return "[??????????]"
    filled = round(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the Book Detection RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N entries (default: all).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip CrossEncoder re-ranking (faster, less accurate).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save full results to a JSON file (e.g. eval/results.json).",
    )
    args = parser.parse_args()

    run_evaluation(
        limit=args.limit,
        no_rerank=args.no_rerank,
        output_path=args.output,
    )
