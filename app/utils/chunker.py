from app.config import get_settings


def chunk_text(text: str) -> list[str]:
    """
    Boundary-aware recursive chunker.

    The old approach: slice every N characters regardless of meaning.
    Problem: cuts mid-sentence, separating a role title from its description,
    or a heading from its content. Retrieval quality tanks.

    The new approach — try natural boundaries in order of preference:
      1. Paragraph breaks (double newline)  — best, preserves complete thoughts
      2. Single newlines                    — good for bullet points / resume lines
      3. Sentence endings (. ! ?)           — fallback for dense prose
      4. Raw character slice                — last resort only

    For each boundary type we ask: "can I split here and keep chunks
    within the size limit?" If yes, we split there. If the resulting
    piece is still too long, we recurse with the next boundary type.

    Overlap is then added by taking the tail of the previous chunk and
    prepending it to the next — so context bleeds across boundaries.

    Why this matters for RAG:
    A resume line like "President, Zeitgeist Club | IIIT Dharwad" followed
    by its bullet description will stay in the same chunk because they share
    a paragraph. The retriever can then find both the title AND context in
    one hit, instead of half the answer being invisible.
    """
    settings = get_settings()
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    if not text.strip():
        return []

    # Step 1: split into raw pieces using the best available boundary
    raw_pieces = _split_on_boundaries(text.strip(), size)

    # Step 2: merge small pieces into chunks that approach `size` characters
    # without exceeding it, then add overlap between adjacent chunks
    return _merge_with_overlap(raw_pieces, size, overlap)


# ── Internal helpers ──────────────────────────────────────────────────────────

# Separators tried in order — most structure-preserving first
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? "]


def _split_on_boundaries(text: str, size: int) -> list[str]:
    """
    Recursively split text using the most natural boundary available.

    If the text fits in one chunk already — return it as-is.
    Otherwise find the best separator that exists in this text, split on it,
    and recurse on any piece that's still too large.
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    # Try each separator in preference order
    for sep in _SEPARATORS:
        if sep in text:
            pieces = text.split(sep)
            result = []
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                if len(piece) <= size:
                    result.append(piece)
                else:
                    # This piece is still too long — recurse with same sep list
                    # but sep is no longer available in piece (we just split on it)
                    result.extend(_split_on_boundaries(piece, size))
            return result

    # No separator found at all — hard character split as absolute last resort
    return [text[i:i+size].strip() for i in range(0, len(text), size) if text[i:i+size].strip()]


def _merge_with_overlap(pieces: list[str], size: int, overlap: int) -> list[str]:
    """
    Merge small pieces into chunks up to `size` characters,
    then prepend `overlap` characters from the previous chunk.

    Why merge? Splitting on newlines might produce very short pieces
    (a single bullet point line). Merging packs them back together into
    dense, meaningful chunks that embed well.

    Why overlap? So that if a role title ends one chunk and its description
    starts the next, the description chunk still begins with the tail of the
    title chunk — keeping them contextually linked.
    """
    if not pieces:
        return []

    chunks = []
    current = ""

    for piece in pieces:
        # If adding this piece keeps us under the size limit, accumulate it
        separator = "\n" if current else ""
        candidate = current + separator + piece

        if len(candidate) <= size:
            current = candidate
        else:
            # Current chunk is full — save it and start fresh
            if current.strip():
                chunks.append(current.strip())
            current = piece

    # Don't forget the last accumulated piece
    if current.strip():
        chunks.append(current.strip())

    # Add overlap: prepend the tail of the previous chunk to the current one
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + "\n" + chunks[i])

    return overlapped