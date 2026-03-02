"""
app/utils/image_utils.py
========================
Image validation, resizing, and base64 encoding utilities.

Used by the detection route (app/api/v1/routes/detection.py) to prepare
raw image bytes received from the HTTP upload for sending to OpenAI.

What this module does:
  1. Opens the image with Pillow to verify it is a valid image file.
  2. Converts palette/RGBA mode images to RGB (JPEG compatibility).
  3. Resizes images exceeding OpenAI's 2048px dimension limit using LANCZOS
     resampling to preserve text readability.
  4. Returns the processed image as a base64 string + confirmed MIME type.

OpenAI vision limits (as of gpt-4o):
  - Max image size: 20MB (we enforce a lower limit via MAX_IMAGE_SIZE_MB).
  - Max dimension for "high" detail: 2048px on the short side.
    Images larger than this are resized by OpenAI anyway, but we resize first
    to reduce upload payload and latency.

Dependencies:
  - Pillow  (pip install Pillow)
"""

import base64
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException


# OpenAI recommends images ≤ 2048px on the longest side for "high" detail.
MAX_DIMENSION = 2048


def validate_and_encode_image(raw_bytes: bytes, media_type: str) -> tuple[str, str]:
    """
    Validate, normalise, and base64-encode image bytes for the OpenAI API.

    Steps performed:
      1. Attempt to open with Pillow (raises HTTP 422 on invalid image data).
      2. Convert RGBA or palette (P) mode to RGB for JPEG/PNG compatibility.
         When converting to RGB, the output format is forced to JPEG.
      3. Resize if the longest dimension exceeds MAX_DIMENSION (2048px).
         Uses LANCZOS resampling for best text legibility after downscale.
      4. Re-encode to bytes and base64-encode for the OpenAI data URI.

    Args:
        raw_bytes:  Raw image bytes from the HTTP upload (UploadFile.read()).
        media_type: MIME type declared by the client ("image/jpeg", "image/png",
                    "image/webp", or "image/gif").

    Returns:
        Tuple of (base64_string, confirmed_media_type).
          - base64_string: Plain base64 string (NOT a data URI).
          - confirmed_media_type: May differ from input if RGBA was converted
            to JPEG (e.g. "image/png" → "image/jpeg").

    Raises:
        HTTPException 422: If raw_bytes cannot be opened as a valid image.
    """
    try:
        img = Image.open(BytesIO(raw_bytes))
        img.verify()           # Raises if file is truncated or corrupted
        img = Image.open(BytesIO(raw_bytes))   # Re-open after verify() consumes stream
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid image file: {exc}",
        ) from exc

    # Convert palette/RGBA to RGB — JPEG does not support transparency.
    if img.mode in ("RGBA", "P", "LA", "L"):
        img = img.convert("RGB")
        media_type = "image/jpeg"

    # Resize if necessary, preserving aspect ratio.
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    # Re-encode to bytes.
    buffer = BytesIO()
    fmt = _pil_format(media_type)
    save_kwargs = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = 92  # High quality to preserve text sharpness

    img.save(buffer, **save_kwargs)
    processed_bytes = buffer.getvalue()

    b64 = base64.standard_b64encode(processed_bytes).decode("utf-8")
    return b64, media_type


def _pil_format(media_type: str) -> str:
    """
    Map a MIME type string to the Pillow save format identifier.

    Args:
        media_type: e.g. "image/jpeg", "image/png", "image/webp".

    Returns:
        Pillow format string: "JPEG", "PNG", or "WEBP".
        Defaults to "JPEG" for unknown types.
    """
    mapping = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
        "image/gif": "PNG",   # Convert GIF to PNG (static frame)
    }
    return mapping.get(media_type, "JPEG")
