import fitz  
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.utils.chunker import chunk_text

# Load embedding model once at module level — expensive to reload
# all-MiniLM-L6-v2 is ~90MB, runs on CPU, good quality for retrieval
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model
  

def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract all text from a PDF using PyMuPDF.
    fitz.open() handles most PDF structures; complex scanned PDFs need OCR (out of scope).
    """
    doc = fitz.open(str(file_path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text  


def ingest_document(file_path: Path, filename: str) -> dict:
    """
    Full ingestion pipeline for one document.
    Returns a summary dict with document_id and chunk_count.

    Steps:
      1. Extract raw text from PDF
      2. Split into overlapping chunks
      3. Embed each chunk into a 384-dim vector
      4. Store vectors + metadata in Qdrant
    """
    settings = get_settings()
    client = get_qdrant_client()
    model = get_embedding_model()

    document_id = str(uuid.uuid4())

    # Step 1 — parse
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text.strip():
        raise ValueError("Could not extract text from PDF. File may be scanned or empty.")

    # Step 2 — chunk
    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError("Document produced no chunks after splitting.")

    # Step 3 — embed
    # encode() returns a numpy array of shape (num_chunks, 384)
    vectors = model.encode(chunks, show_progress_bar=False).tolist()

    # Step 4 — store in Qdrant
    # Each point = one chunk vector + metadata payload
    # The payload is what gets returned alongside search results
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "document_id": document_id,
                "filename": filename,
                "chunk_text": chunks[i],
                "chunk_index": i,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        for i in range(len(chunks))
    ]

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "chunk_count": len(chunks),
    }
