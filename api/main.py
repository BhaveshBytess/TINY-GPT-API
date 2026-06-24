"""
TinyGPT API — main application.

A GPT-style language model served over HTTP, with cloud LLM fallback,
structured logging, request tracing, and global error handling.
"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    EchoRequest, EchoResponse,
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse,
)
from api.inference import model_inference
from api.cloud_client import cloud_client, CloudAPIError
from api.logging_config import setup_logging
from api.middleware import RequestContextMiddleware
from api.exceptions import register_exception_handlers

import asyncio
from fastapi.responses import StreamingResponse

from fastapi.staticfiles import StaticFiles
from api.history import prompt_history

# ─────────────────────────────────────────
#  Configure logging FIRST — before anything else runs
# ─────────────────────────────────────────
setup_logging(level="INFO")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "model" / "weights" / "tiny_gpt.pt"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "tinyshakespeare.txt"


# ─────────────────────────────────────────
#  Lifespan: startup and shutdown
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    # Phase 1: TinyGPT
    try:
        model_inference.load(MODEL_WEIGHTS_PATH, MODEL_DATA_PATH)
        logger.info("TinyGPT loaded")
    except Exception as e:
        logger.warning("TinyGPT not loaded: %s", e)

    # Phase 2: embedding model for RAG
    try:
        embedding_model.load()
        logger.info("Embedding model loaded for RAG (store has %d chunks)",
                    vector_store.count())
    except Exception as e:
        logger.error("Embedding model failed to load: %s", e)

    # cloud client check (unchanged)
    if cloud_client.is_configured():
        logger.info("Cloud client configured: %s", cloud_client.model)
    else:
        logger.warning("Cloud client not configured")

    yield
    logger.info("Shutting down...")


# ─────────────────────────────────────────
#  Create the app
# ─────────────────────────────────────────
app = FastAPI(
    title="TinyGPT API",
    description="A GPT-style language model built from scratch, with cloud LLM fallback.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────
#  Middleware — order matters: CORS first, then RequestContext
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Production: list specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)


# ─────────────────────────────────────────
#  Global exception handlers
# ─────────────────────────────────────────
register_exception_handlers(app)

app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

# ─────────────────────────────────────────
#  Endpoint 1: Health Check
# ─────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check for load balancers and monitoring."""
    return {
        "status": "ok",
        "model_loaded": model_inference.model is not None,
        "cloud_configured": cloud_client.is_configured(),
    }


# ─────────────────────────────────────────
#  Endpoint 2: Echo
# ─────────────────────────────────────────
@app.post("/echo", response_model=EchoResponse)
def echo(request: EchoRequest):
    """Echo back the user's message. Learning endpoint for request/response flow."""
    return EchoResponse(response=f"You said: {request.message}")


# ─────────────────────────────────────────
#  Endpoint 3: Generate (local TinyGPT only)
# ─────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    """Generate text using the local TinyGPT model."""
    if model_inference.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    start_time = time.time()

    try:
        generated = model_inference.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        # Let the generic handler catch unexpected errors, but wrap
        # known generation issues clearly
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )

    elapsed = time.time() - start_time
    tokens_generated = len(generated) - len(request.prompt)

    logger.info(
        "Generate completed: tokens=%d latency=%dms prompt_len=%d",
        tokens_generated, int(elapsed * 1000), len(request.prompt),
    )

    return GenerateResponse(
        generated_text=generated,
        tokens_generated=tokens_generated,
    )


# ─────────────────────────────────────────
#  Endpoint 4: Chat (routes between TinyGPT and cloud)
# ─────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint routing between local TinyGPT and cloud LLMs.

    Note: no try/except for CloudAPIError needed here — the global
    exception handler registered via register_exception_handlers()
    catches it and returns 503 automatically.
    """
    start_time = time.time()

    if request.model == "tiny":
        if model_inference.model is None:
            raise HTTPException(
                status_code=503,
                detail="TinyGPT is not loaded. Check server logs.",
            )
        response_text = model_inference.generate(
            prompt=request.message,
            max_tokens=request.max_tokens,
        )
        model_used = "tiny-gpt"

    elif request.model == "cloud":
        if not cloud_client.is_configured():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Cloud provider '{cloud_client.provider}' is not configured. "
                    f"Set the appropriate API key in .env"
                ),
            )
        # CloudAPIError raised here is caught by the global handler → 503
        response_text = await cloud_client.generate(
            prompt=request.message,
            max_tokens=request.max_tokens,
        )
        model_used = f"{cloud_client.provider}:{cloud_client.model}"

    latency_ms = int((time.time() - start_time) * 1000)

    logger.info(
        "Chat completed: model=%s latency=%dms message_len=%d",
        model_used, latency_ms, len(request.message),
    )

    prompt_history.add(message=request.message, model=request.model)

    return ChatResponse(
        response=response_text,
        model_used=model_used,
        latency_ms=latency_ms,
    )


@app.get("/history")
def get_history():
    """Return the last N prompts (in-memory, resets on restart)."""
    return {"history": prompt_history.get_all()}


@app.delete("/history")
def clear_history():
    """Clear the prompt history."""
    prompt_history.clear()
    return {"status": "cleared"}


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Stream generated tokens one at a time using Server-Sent Events.
    The client receives 'data: <token>\\n\\n' chunks as they're produced.
    """
    if model_inference.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    async def event_generator():
        try:
            for token in model_inference.generate_stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                # SSE format: each event is "data: <content>\n\n"
                yield f"data: {token}\n\n"
                await asyncio.sleep(0)   # yield control to event loop
            # Signal completion
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Streaming failed")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


import time
from fastapi import HTTPException
from rag.pipeline import answer_question
from rag.embeddings import embedding_model
from rag.vector_store import vector_store
from api.schemas import RAGRequest, RAGResponse, RAGSource


@app.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    Answer a question using Retrieval-Augmented Generation.

    Retrieves relevant chunks from the knowledge base, builds a grounded
    prompt, and returns the LLM's answer with source citations.
    """
    # Guard: is anything ingested?
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is empty. Ingest documents first via /ingest.",
        )

    start = time.time()

    # CloudAPIError here is caught by the global handler → 503
    result = await answer_question(
        question=request.question,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        max_tokens=request.max_tokens,
    )

    latency_ms = int((time.time() - start) * 1000)

    logger.info(
        "RAG query: grounded=%s chunks=%d latency=%dms q='%s'",
        result["grounded"], result["num_chunks_used"],
        latency_ms, request.question[:50],
    )

    return RAGResponse(
        answer=result["answer"],
        sources=[RAGSource(**s) for s in result["sources"]],
        num_chunks_used=result["num_chunks_used"],
        grounded=result["grounded"],
        latency_ms=latency_ms,
    )


from rag.ingest import ingest_documents
from api.schemas import IngestRequest, IngestResponse


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Ingest documents into the knowledge base.

    Note: this is a synchronous, CPU-bound operation (embedding). For large
    ingestion jobs, production would offload this to a background worker
    rather than blocking the request.
    """
    # Convert Pydantic models to the dict format the pipeline expects
    docs = [{"text": d.text, "source": d.source} for d in request.documents]

    summary = ingest_documents(docs)

    logger.info(
        "Ingest: %d docs → %d chunks",
        summary["documents"], summary["chunks"],
    )

    return IngestResponse(
        documents_processed=summary["documents"],
        chunks_created=summary["chunks"],
        total_chunks_in_store=vector_store.count(),
    )
