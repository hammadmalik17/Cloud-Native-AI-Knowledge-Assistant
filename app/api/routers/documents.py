import uuid
import aiofiles
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.services.ingestor import ingest_document
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DeleteResponse,
    IngestionStatus,
)
from app.db.qdrant import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory store for document metadata
# In production: replace with PostgreSQL or Redis
# This is a deliberate simplification — we'll note it as a known limitation
_document_registry: dict[str, dict] = {}


def _run_ingestion(file_path: Path, filename: str, document_id: str) -> None:
    """Background task — runs after HTTP response is already sent."""
    try:
        result = ingest_document(file_path, filename)
        _document_registry[document_id].update({
            "status": IngestionStatus.done,
            "chunk_count": result["chunk_count"],
        })
    except Exception as e:
        _document_registry[document_id].update({
            "status": IngestionStatus.failed,
            "error": str(e),
        })


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept a PDF, save it, kick off ingestion in the background.
    Returns immediately with 202 — don't make the client wait for embedding.
    """
    settings = get_settings()

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {settings.max_file_size_mb}MB.",
        )

    # Save to disk
    document_id = str(uuid.uuid4())
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / f"{document_id}.pdf"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)

    # Register document as "processing"
    _document_registry[document_id] = {
        "document_id": document_id,
        "filename": file.filename,
        "status": IngestionStatus.processing,
        "chunk_count": 0,
        "error": None,
        "created_at": datetime.now(timezone.utc),
    }

    # Kick off ingestion without blocking the response
    background_tasks.add_task(_run_ingestion, file_path, file.filename, document_id)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status=IngestionStatus.processing,
    )


@router.get("", response_model=list[DocumentStatusResponse])
async def list_documents():
    """Return all documents and their ingestion status."""
    return [DocumentStatusResponse(**doc) for doc in _document_registry.values()]


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Poll ingestion status for a specific document."""
    doc = _document_registry.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentStatusResponse(**doc)


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """
    Delete a document's vectors from Qdrant and remove from registry.
    Uses payload filtering to find all chunks belonging to this document_id.
    """
    if document_id not in _document_registry:
        raise HTTPException(status_code=404, detail="Document not found.")

    settings = get_settings()
    client = get_qdrant_client()

    # Delete all vectors whose payload.document_id matches
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )

    chunk_count = _document_registry[document_id].get("chunk_count", 0)
    del _document_registry[document_id]

    return DeleteResponse(deleted=True, chunks_removed=chunk_count)
