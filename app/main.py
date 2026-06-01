import time
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import get_settings
from app.db.qdrant import ensure_collection_exists
from app.api.routers import documents, query
from app.models.schemas import HealthResponse
from app.db.qdrant import get_qdrant_client

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup before serving requests.
    This is where we establish connections and warm up models.

    Using lifespan instead of @app.on_event("startup") — the modern FastAPI pattern.
    """
    print("Starting RAG assistant...")
    ensure_collection_exists()  # create Qdrant collection if it doesn't exist
    print("Ready.")
    yield
    # Anything after yield runs on shutdown — close connections here if needed


app = FastAPI(
    title="RAG Knowledge Assistant",
    description="Upload documents, ask questions, get grounded answers.",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """
    Liveness endpoint — checked by Docker, deployment platforms, and load balancers.
    Checks both app health AND vector DB reachability.
    """
    settings = get_settings()
    vector_db_status = "up"

    try:
        client = get_qdrant_client()
        client.get_collections()  # lightweight ping
    except Exception:
        vector_db_status = "down"

    overall = "ok" if vector_db_status == "up" else "degraded"

    return HealthResponse(
        status=overall,
        vector_db=vector_db_status,
        version=app.version,
        uptime_s=int(time.time() - _start_time),
    )
