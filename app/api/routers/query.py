import time
from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse
from app.services.retriever import retrieve_relevant_chunks
from app.services.generator import generate_answer

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    The core RAG endpoint.

    Flow: embed question → retrieve top-K chunks → build prompt → call Gemini → return answer + sources

    We return sources so the user can verify the answer.
    We track latency so we can observe performance over time.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()

    # Step 1: Retrieve relevant chunks via vector similarity
    chunks = retrieve_relevant_chunks(
        question=request.question,
        top_k=request.top_k,
    )

    # Step 2: Generate grounded answer from retrieved context
    answer = generate_answer(
        question=request.question,
        chunks=chunks,
    )

    latency_ms = int((time.time() - start) * 1000)

    return QueryResponse(
        answer=answer,
        sources=chunks,
        latency_ms=latency_ms,
    )
