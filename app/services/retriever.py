from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.services.ingestor import get_embedding_model
from app.models.schemas import SourceChunk


def retrieve_relevant_chunks(question: str, top_k: int = 5) -> list[SourceChunk]:
    """
    Embeds the question using the same model used at ingestion time,
    then performs approximate nearest-neighbor search in Qdrant.

    IMPORTANT: you MUST use the same embedding model here as in ingestor.py.
    If you change models, all stored vectors become meaningless — you'd need
    to re-ingest every document. This is a real operational concern in production.
    """
    settings = get_settings()
    client = get_qdrant_client()
    model = get_embedding_model()

    # Embed the question — produces a single 384-dim vector
    query_vector = model.encode(question).tolist()

    # Search Qdrant — returns top_k closest vectors by cosine similarity
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,  # we want the chunk_text and metadata back
    )

    # Map Qdrant results to our schema
    chunks = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append(
            SourceChunk(
                chunk_text=payload.get("chunk_text", ""),
                document_id=payload.get("document_id", ""),
                filename=payload.get("filename", ""),
                score=round(hit.score, 4),
            )
        )

    return chunks
