from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash-lite"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_documents"

    # Embedding model (local, no API key)
    embedding_model: str = "all-MiniLM-L6-v2"

    # File upload
    upload_dir: str = "uploads"
    max_file_size_mb: int = 20

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# lru_cache means this is only created once — not on every request
@lru_cache()
def get_settings() -> Settings:
    return Settings()