from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from functools import lru_cache
from app.config import get_settings

# Embedding dimension for all-MiniLM-L6-v2

EMBEDDING_DIM = 384


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection_exists() -> None:
    """
    Called at app startup.
    Creates the Qdrant collection if it doesn't already exist.
    Idempotent — safe to call multiple times.
    """
    client = get_qdrant_client()
    settings = get_settings()
    existing = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,   # cosine similarity for text embeddings
            ),
        )
        print(f"Created Qdrant collection: {settings.qdrant_collection}")
    else:
        print(f"Qdrant collection already exists: {settings.qdrant_collection}")
