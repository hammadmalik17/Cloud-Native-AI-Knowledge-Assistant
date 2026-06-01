import google.generativeai as genai
from app.config import get_settings
from app.models.schemas import SourceChunk

_gemini_configured = False


def _configure_gemini() -> None:
    global _gemini_configured
    if not _gemini_configured:
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        _gemini_configured = True


def build_prompt(question: str, chunks: list[SourceChunk]) -> str:
    """
    Constructs the RAG prompt — the most important design decision in this file.

    The prompt has three parts:
      1. System instruction — tells the LLM to stay grounded in context
      2. Retrieved context — the chunks we pulled from Qdrant
      3. The user's question

    Why this structure? Without the instruction, the LLM uses its training data
    to hallucinate an answer. With it, it's forced to reason only from what we provide.
    """
    context_blocks = "\n\n---\n\n".join(
        f"[Source: {c.filename}, score: {c.score}]\n{c.chunk_text}"
        for c in chunks
    )

    return f"""You are a precise knowledge assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer this."
Do not use your general training knowledge. Cite which document your answer comes from.

CONTEXT:
{context_blocks}

QUESTION: {question}

ANSWER:"""


def generate_answer(question: str, chunks: list[SourceChunk]) -> str:
    """
    Sends the grounded prompt to Gemini and returns the answer.
    Falls back gracefully if no chunks were retrieved.
    """
    _configure_gemini()
    settings = get_settings()

    if not chunks:
        return "No relevant content found in uploaded documents. Please upload documents first."

    prompt = build_prompt(question, chunks)
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(prompt)

    return response.text
