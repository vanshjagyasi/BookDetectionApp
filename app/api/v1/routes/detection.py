"""
app/api/v1/routes/detection.py
================================
The single public endpoint: POST /api/v1/detect-book

Orchestrates the full pipeline:
  1. Validate uploaded image (type, size).
  2. Encode image to base64.
  3. Call VisionService  → VisionExtraction
  4. Call RAGService     → candidate books + context string
  5. Call LLMService     → BookInfo (structured JSON)
  6. Return DetectionResponse

HTTP behaviour:
  POST /api/v1/detect-book
    Content-Type: multipart/form-data
    Body: file=<image file>
  → 200 DetectionResponse JSON
  → 413 if image > MAX_IMAGE_SIZE_MB
  → 415 if file type is not JPEG/PNG/WEBP/GIF
  → 422 if image bytes are corrupt/unreadable
  → 500 on unexpected errors (OpenAI API failures, etc.)

For full curl examples, see docs/API_REFERENCE.md.
"""

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
#from langfuse import get_client, observe

from app.config import Settings, get_settings
from app.db.vector_store import VectorStore
from app.dependencies import get_reranker, get_vector_store
from app.schemas.book import DetectionResponse
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.vision_service import VisionService
from app.utils.image_utils import MAX_DIMENSION, validate_and_encode_image

router = APIRouter(tags=["Book Detection"])

# Map client Content-Type → normalised MIME type passed to OpenAI.
# Keys include both "image/jpg" and "image/jpeg" for browser compatibility.
SUPPORTED_MEDIA_TYPES: dict[str, str] = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
    "image/gif": "image/gif",
}


@router.post(
    "/detect-book",
    response_model=DetectionResponse,
    summary="Identify a book from its cover image",
    description=(
        "Upload a photo of a book cover (front, back, or side/spine). "
        "The API extracts visible text using GPT-4o vision, queries a vector "
        "database of known books, then returns structured JSON with full metadata. "
        "Accepted formats: JPEG, PNG, WEBP, GIF. Maximum size: controlled by "
        "MAX_IMAGE_SIZE_MB environment variable."
    ),
    responses={
        200: {"description": "Book identified successfully."},
        413: {"description": "Image file exceeds the maximum allowed size."},
        415: {"description": "Unsupported image format."},
        422: {"description": "Image file is corrupt or cannot be decoded."},
        500: {"description": "Internal error (AI API failure, etc.)."},
    },
)
#@observe(name="detect-book")   # Parent Langfuse trace — all service spans become children (disabled for debugging)
async def detect_book(
    file: UploadFile = File(
        ...,
        description="Book cover image. Accepted formats: JPEG, PNG, WEBP, GIF.",
    ),
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store),
    reranker=Depends(get_reranker),
) -> DetectionResponse:
    """
    Identify a book from a cover image using the multimodal RAG pipeline.

    Pipeline steps:
      1. Validate file MIME type and size.
      2. Read and process image bytes (resize if needed, base64-encode).
      3. Vision extraction: GPT-4o reads the cover and returns VisionExtraction.
      4. RAG retrieval: embed extraction text, query ChromaDB for top-k candidates.
      5. LLM synthesis: GPT-4o cross-references image + RAG context → BookInfo.
      6. Return DetectionResponse with BookInfo and extraction notes.

    Args:
        file:         Uploaded image file (multipart/form-data).
        settings:     App config (injected via Depends).
        vector_store: ChromaDB + embedding model (injected via Depends).

    Returns:
        DetectionResponse containing the identified BookInfo.

    Raises:
        HTTPException 415: Unsupported image format.
        HTTPException 413: Image exceeds size limit.
        HTTPException 422: Corrupt image file.
        HTTPException 500: AI API or internal error.
    """
    # --- Validate MIME type ---
    content_type = (file.content_type or "").lower()
    if content_type not in SUPPORTED_MEDIA_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported image type: {content_type!r}. "
                f"Supported: {', '.join(sorted(set(SUPPORTED_MEDIA_TYPES.values())))}"
            ),
        )
    media_type = SUPPORTED_MEDIA_TYPES[content_type]

    # --- Read and validate file size ---
    raw_bytes = await file.read()
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image size {len(raw_bytes) / 1024 / 1024:.1f}MB exceeds "
                f"the {settings.MAX_IMAGE_SIZE_MB}MB limit."
            ),
        )

    # --- Process image: validate, resize, encode ---
    image_b64, media_type = validate_and_encode_image(raw_bytes, media_type)
    # Estimate image size from base64 length (base64 is ~4/3 of original bytes)
    image_size_kb = round(len(image_b64) * 3 / 4 / 1024)
    print(f"[DEBUG] image encoded — {image_size_kb}KB, type={media_type}, max_dim={MAX_DIMENSION}px", flush=True)

    # --- Initialise services ---
    vision_svc = VisionService(settings)
    rag_svc = RAGService(vector_store, reranker)
    llm_svc = LLMService(settings)

    # --- Stage 1: Vision extraction ---
    # asyncio.to_thread offloads sync/blocking calls to a thread pool so that
    # langfuse.openai's internal OTel instrumentation (which uses asyncio)
    # does not deadlock against the already-running uvicorn event loop.
    print(f"[DEBUG] stage 1 — vision extraction (model={settings.OPENAI_MODEL}, detail={settings.VISION_DETAIL})...", flush=True)
    extraction = await asyncio.to_thread(
        vision_svc.extract_book_text, image_b64, media_type
    )
    print(f"[DEBUG] stage 1 done — title={extraction.visible_title!r} author={extraction.visible_author!r}", flush=True)

    # --- Stage 2: RAG retrieval ---
    print(f"[DEBUG] stage 2 — RAG retrieval (fetch_k={settings.RAG_FETCH_K}, rerank={'on' if reranker else 'off'})...", flush=True)
    rag_results = await asyncio.to_thread(rag_svc.retrieve, extraction)
    rag_context = rag_svc.format_context(rag_results)
    print(f"[DEBUG] stage 2 done — {len(rag_results)} candidate(s)", flush=True)

    # --- Stage 3: LLM synthesis ---
    print(f"[DEBUG] stage 3 — LLM synthesis (model={settings.OPENAI_SYNTHESIS_MODEL})...", flush=True)
    book_info = await asyncio.to_thread(
        llm_svc.generate_book_info, extraction, rag_context
    )
    print(f"[DEBUG] stage 3 done — title={book_info.title!r} confidence={book_info.confidence_score}", flush=True)

    # --- Log trace output to Langfuse ---
    #print("[DEBUG] langfuse — updating trace...", flush=True)
    #get_client().update_current_trace(
    #    output={"title": book_info.title, "confidence_score": book_info.confidence_score},
    #    metadata={"filename": file.filename, "model": settings.OPENAI_MODEL},
    #)
    #print("[DEBUG] langfuse — trace updated", flush=True)

    # --- Build response ---
    notes_parts = []
    if extraction.visible_title:
        notes_parts.append(f"Vision read: '{extraction.visible_title}'")
    if extraction.visible_author:
        notes_parts.append(f"Author: '{extraction.visible_author}'")
    rerank_note = f"re-ranked {settings.RAG_FETCH_K} → {len(rag_results)}" if reranker else f"matched {len(rag_results)}"
    notes_parts.append(f"RAG {rerank_note} candidate(s)")

    return DetectionResponse(
        success=True,
        book=book_info,
        extraction_notes=" | ".join(notes_parts),
    )
