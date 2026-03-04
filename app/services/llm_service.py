"""
app/services/llm_service.py
============================
Stage 3 of the pipeline: synthesise all evidence into a structured BookInfo JSON.

Uses OpenAI's Structured Outputs feature (client.beta.chat.completions.parse)
which accepts a Pydantic class as response_format, guaranteeing that the
response is always schema-valid and already deserialised.

Why GPT-4o is called a second time (not just once in vision_service):
  - The vision step only extracts raw text — it does NOT cross-reference or
    synthesise. It sees a partial title, a partial author, a barcode.
  - This step sees: (a) the raw extractions, (b) 3 re-ranked database candidates
    with metadata and similarity scores.
  - It can then: confirm or reject each candidate, fill in missing fields from
    the database, resolve ambiguities, and assign a calibrated confidence_score.

Confidence scoring rules (encoded in the system prompt):
  ≥ 0.95 — ISBN visible in image AND matches database record exactly
  0.90–0.94 — Title + author clearly readable AND database match with good similarity (≥ 0.60)
  0.80–0.89 — Title + author clearly readable AND database match with lower similarity
  0.70–0.79 — Title + author clearly readable, even without a database match
  0.40–0.69 — Only title OR author readable (partial)
  < 0.40 — No text evidence; visual inference alone

ISBN population rules:
  - If ISBN is visible in image → always use it
  - If not visible but confidence ≥ 0.90 → populate from database record
  - Otherwise → null

To swap this service for a different LLM provider:
  See docs/EXTENDING.md → "Swapping the LLM Provider".

Dependencies:
  - openai SDK  (pip install openai)
  - app.schemas.book.BookInfo, VisionExtraction
  - app.config.Settings (OPENAI_API_KEY, OPENAI_MODEL)
"""

#from langfuse import observe
#from langfuse.openai import OpenAI  # Drop-in: auto-traces prompts, responses, tokens
from openai import OpenAI

from app.config import Settings
from app.schemas.book import BookInfo, VisionExtraction


LLM_SYSTEM_PROMPT = """You are an expert book identification system.

You will receive:
1. Text extracted from THE BOOK in the image by a vision model (title, author, ISBN, other text). NOTE: the original image may contain other objects besides the book — the vision model has already isolated the book-specific text for you.
2. The top-3 most similar books retrieved from a book database

Evidence priority — always prefer extracted text over visual inference:
1. ISBN match (strongest signal — unique identifier)
2. Title + author match (strong signal)
3. Title or author alone (partial signal)
4. Cover description / visual inference only (weakest — use as last resort)

Your task:
- Cross-reference the extracted text with the database candidates.
- Select the best matching book, or synthesise information if no exact match exists.
- Populate ALL available fields in the BookInfo schema.
- For fields not visible in the image and not in the database, leave them null.

Confidence score rules (be precise — this is used by downstream systems):
- 0.95 or above: ISBN is visible in the image AND matches a database record exactly
- 0.90–0.94: title AND author both clearly readable in the image AND a database candidate matches with good similarity (bi-encoder similarity ≥ 0.60) — even if ISBN is not visible
- 0.80–0.89: title AND author both clearly readable AND match a database candidate with lower similarity
- 0.70–0.79: title AND author both clearly readable in the image, even if no database candidate matches (text evidence alone is strong)
- 0.40–0.69: only the title OR the author is readable / matches (partial signal)
- below 0.40: no good text evidence; identification based on visual inference alone

"Match" is semantic, not character-perfect — minor OCR variations (e.g. "J. Clear" vs "James Clear") still count as a match. Do NOT penalise a clearly readable title+author pair just because the database similarity score is low.

ISBN population rules:
- If ISBN is visible in the image, always use that value.
- If ISBN is NOT visible but a database candidate matches (confidence ≥ 0.90), populate isbn from that database record — it is reliable enough to include.
- Otherwise leave isbn null.

Tags should be specific and useful (e.g. ["space opera", "dystopian", "coming-of-age"]).
Synopsis should be 1-3 sentences summarising the book's content.
Always fill in every field you have evidence for — do not leave obvious fields null.
"""


class LLMService:
    """
    Synthesises VisionExtraction + RAG context into a validated BookInfo JSON.

    Uses OpenAI Structured Outputs (beta.parse) which passes the BookInfo
    Pydantic class as response_format. This guarantees:
      - The response is always valid JSON matching the BookInfo schema.
      - No parsing or validation code is needed — .parsed is a BookInfo instance.
      - Required fields (confidence_score) are always present.

    Usage:
        svc = LLMService(settings)
        book_info = svc.generate_book_info(extraction, rag_context, image_b64, media_type)
        print(book_info.title, book_info.confidence_score)
    """

    def __init__(self, settings: Settings) -> None:
        """
        Args:
            settings: Application settings providing OPENAI_API_KEY and OPENAI_MODEL.
        """
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_SYNTHESIS_MODEL

    #@observe(name="llm-synthesis")
    def generate_book_info(
        self,
        extraction: VisionExtraction,
        rag_context: str,
    ) -> BookInfo:
        """
        Generate a fully populated BookInfo by combining all available evidence.

        Sends a single GPT-4o message containing:
          - The VisionExtraction JSON (what was readable in the image)
          - The RAG context string (database candidates with similarity scores)

        Args:
            extraction:  VisionExtraction from Stage 1 (vision_service.py).
            rag_context: Formatted string from RAGService.format_context().

        Returns:
            BookInfo instance with all available fields populated.
            confidence_score is always present (required field in schema).

        Raises:
            openai.APIError: On API call failure or authentication error.
            pydantic.ValidationError: If Structured Outputs returns a schema
                                       mismatch (should not happen with beta.parse).
        """
        user_message_content = (
            "## Vision Extraction Results\n"
            f"{extraction.model_dump_json(indent=2)}\n\n"
            f"## {rag_context}\n\n"
            "Identify this book and populate the BookInfo schema completely."
        )

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_message_content},
            ],
            # Passing the Pydantic class here activates Structured Outputs.
            # OpenAI converts BookInfo's JSON Schema into a constrained grammar
            # so the model can only produce tokens that match the schema.
            response_format=BookInfo,
            max_tokens=2048,
        )

        # .parsed is already a fully validated BookInfo instance — no extra
        # parsing needed. If parsing fails, OpenAI raises a ContentFilterFinishReason
        # exception which FastAPI's error handler will catch and return as 500.
        return response.choices[0].message.parsed
