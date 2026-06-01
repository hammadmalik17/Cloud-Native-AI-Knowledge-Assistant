from app.config import get_settings


def chunk_text(text: str) -> list[str]:
    """
    Splits text into overlapping chunks.

    Why overlap? If a sentence spans a chunk boundary, overlap ensures
    neither chunk loses context. Typical values: 500 tokens, 50 overlap.

    This is a simple character-based splitter. In production you'd use
    token-based splitting to match the embedding model's context window exactly.
    """
    settings = get_settings()
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    if not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap  

    return chunks
