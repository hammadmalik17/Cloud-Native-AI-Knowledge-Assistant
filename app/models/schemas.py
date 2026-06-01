from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class IngestionStatus(str, Enum):
    processing = "processing"
    done = "done"
    failed = "failed"


# --- Document models ---

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: IngestionStatus


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: IngestionStatus
    chunk_count: int = 0
    error: Optional[str] = None
    created_at: datetime


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    status: IngestionStatus
    chunk_count: int
    created_at: datetime


class DeleteResponse(BaseModel):
    deleted: bool
    chunks_removed: int


# --- Query models ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection_name: Optional[str] = None


class SourceChunk(BaseModel):
    chunk_text: str
    document_id: str
    filename: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: int


# --- Health model ---

class HealthResponse(BaseModel):
    status: str           # "ok" or "degraded"
    vector_db: str        # "up" or "down"
    version: str
    uptime_s: int
