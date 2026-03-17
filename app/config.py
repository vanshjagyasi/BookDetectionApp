"""
config.py
=========
Centralised application configuration loaded from environment variables / .env file.

All settings are read once at startup via get_settings() which is cached with
@lru_cache so the .env file is only parsed once per process lifetime.

Usage in other modules:
    from app.config import get_settings
    settings = get_settings()
    api_key = settings.OPENAI_API_KEY

Adding a new setting:
    1. Add a typed field to the Settings class below.
    2. Add the variable (with a comment) to .env.example.
    3. Reference it via get_settings() wherever needed.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration.

    All values are read from environment variables.
    If a .env file exists in the working directory it is loaded automatically.
    Fields without defaults are REQUIRED — the app will fail to start if they
    are missing from the environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str
    """OpenAI API key. Required for both vision extraction and JSON generation."""

    OPENAI_MODEL: str = "gpt-4o"
    """GPT model used for vision extraction. gpt-4o supports vision + structured outputs."""

    OPENAI_SYNTHESIS_MODEL: str = "gpt-4.1-nano"
    """GPT model used for Stage 3 synthesis. gpt-4.1-nano is the fastest/cheapest model with structured output support."""

    VISION_DETAIL: str = "auto"
    """OpenAI vision detail level: "high", "low", or "auto". "auto" lets the API choose based on image size."""

    # ------------------------------------------------------------------
    # Google Books API (only used by scripts/populate_db.py)
    # ------------------------------------------------------------------
    GOOGLE_BOOKS_API_KEY: str = ""
    """Google Books API key. Only required when running scripts/populate_db.py."""

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    """Filesystem path where ChromaDB stores its persistent collection data."""

    CHROMA_COLLECTION_NAME: str = "books"
    """Name of the ChromaDB collection that holds book documents + embeddings."""

    # ------------------------------------------------------------------
    # Embedding model (OpenAI API)
    # ------------------------------------------------------------------
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    """
    OpenAI embedding model for converting text into vectors.
    text-embedding-3-small: 1536 dimensions, ~$0.02/M tokens.
    Must stay consistent — changing this after DB population invalidates
    all stored embeddings and requires re-running populate_db.py.
    """

    RAG_TOP_K: int = 5
    """Number of nearest-neighbour books to retrieve from ChromaDB per query."""

    # ------------------------------------------------------------------
    # Langfuse LLM Observability
    # ------------------------------------------------------------------
    LANGFUSE_PUBLIC_KEY: str = ""
    """
    Langfuse project public key. Get from https://cloud.langfuse.com
    Leave empty to disable tracing (the @observe decorator is a no-op
    when no keys are configured).
    """

    LANGFUSE_SECRET_KEY: str = ""
    """Langfuse project secret key. Required alongside LANGFUSE_PUBLIC_KEY."""

    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    """
    Langfuse server URL. Use the default for Langfuse Cloud.
    Change to your own host URL for self-hosted Langfuse.
    """

    # ------------------------------------------------------------------
    # API behaviour
    # ------------------------------------------------------------------
    MAX_IMAGE_SIZE_MB: int = 20
    """Maximum allowed image upload size in megabytes."""

    ALLOWED_ORIGINS: list[str] = ["*"]
    """CORS allowed origins. Restrict in production (e.g. your frontend domain)."""


@lru_cache
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    The @lru_cache decorator ensures .env is parsed exactly once per process.
    Call get_settings() freely — it has no I/O cost after the first call.

    Returns:
        Settings: The populated configuration object.
    """
    return Settings()
