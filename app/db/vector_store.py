"""
app/db/vector_store.py
======================
ChromaDB vector store wrapper with OpenAI embeddings.

Responsibilities:
  - Initialise a persistent ChromaDB client at application startup (via lifespan).
  - Call the OpenAI Embeddings API to convert text into vectors.
  - Expose upsert() for adding/updating book documents during DB population.
  - Expose query() for semantic nearest-neighbour search during inference.

ChromaDB collection configuration:
  - Distance metric: cosine  (correct for normalised OpenAI embedding vectors)
  - Embedding model: text-embedding-3-small  (1536-dimensional vectors)
  - Persistence: disk-backed via CHROMA_PERSIST_DIR (survives restarts)

IMPORTANT — Architecture invariant:
  VectorStore is initialised ONCE inside app/main.py's lifespan() context manager
  and stored in app.state.vector_store.  Never instantiate VectorStore per-request.

Dependencies:
  - chromadb  (pip install chromadb)
  - openai    (pip install openai)
  - app.config.Settings
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from app.config import Settings


class VectorStore:
    """
    Manages the ChromaDB collection and OpenAI embedding API calls.

    Lifecycle:
        store = VectorStore(settings)
        await store.initialize()   # called once in main.py lifespan
        results = store.query("Harry Potter wizard school")
        store.upsert("isbn-123", "Harry Potter ...", {"title": "Harry Potter"})
    """

    def __init__(self, settings: Settings) -> None:
        """
        Store settings reference. Does NOT load ChromaDB yet —
        that happens in initialize() so it can be called async-safely.

        Args:
            settings: Application settings (provides CHROMA_PERSIST_DIR,
                      CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, OPENAI_API_KEY).
        """
        self.settings = settings
        self.client: chromadb.PersistentClient | None = None
        self.collection: chromadb.Collection | None = None
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def initialize(self) -> None:
        """
        Load ChromaDB into memory.

        Must be called once before any query() or upsert() calls.
        Called automatically by app/main.py's lifespan context manager.

        Side effects:
            - Opens (or creates) the ChromaDB persistent directory.
        """
        self.client = chromadb.PersistentClient(
            path=self.settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # hnsw:space=cosine is the correct metric for OpenAI embedding vectors.
        self.collection = self.client.get_or_create_collection(
            name=self.settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, text: str) -> list[float]:
        """
        Convert a text string into an embedding vector via OpenAI API.

        Args:
            text: Any UTF-8 string (title, synopsis, query string, etc.).

        Returns:
            List of floats representing the text in semantic space.

        Raises:
            openai.APIError: On API call failure or authentication error.
        """
        response = self.openai_client.embeddings.create(
            model=self.settings.EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def upsert(self, book_id: str, document: str, metadata: dict) -> None:
        """
        Insert or update a book document in the ChromaDB collection.

        If a document with book_id already exists, it is overwritten — this
        makes populate_db.py safe to re-run without creating duplicates.

        Args:
            book_id:  Unique stable identifier (ISBN-13 preferred, or md5 hash).
            document: Rich text string that will be embedded for similarity search.
                      Format: "{title} by {author}. {description}"
            metadata: Dict of book fields stored alongside the embedding.

        Raises:
            RuntimeError: If initialize() has not been called yet.
        """
        if self.collection is None:
            raise RuntimeError("VectorStore.initialize() must be called before upsert().")
        embedding = self.embed(document)
        self.collection.upsert(
            ids=[book_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    def query(self, query_text: str, n_results: int | None = None) -> list[dict]:
        """
        Find the most semantically similar books to the query string.

        Args:
            query_text: Natural language query built from VisionExtraction fields.
            n_results:  Number of candidates to return. Defaults to RAG_TOP_K.

        Returns:
            List of dicts sorted by distance ascending (most similar first).
            Returns empty list if the collection is empty.

        Raises:
            RuntimeError: If initialize() has not been called yet.
        """
        if self.collection is None:
            raise RuntimeError("VectorStore.initialize() must be called before query().")

        k = n_results or self.settings.RAG_TOP_K
        count = self.collection.count()
        if count == 0:
            return []

        # Clamp k to actual collection size to avoid ChromaDB error
        k = min(k, count)

        embedding = self.embed(query_text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        books = []
        for i in range(len(results["ids"][0])):
            books.append(
                {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return books

    def count(self) -> int:
        """Return the total number of book documents stored in the collection."""
        if self.collection is None:
            return 0
        return self.collection.count()
