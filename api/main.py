from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import time
from pathlib import Path

from api.schemas import (
    EchoRequest, EchoResponse,
    GenerateRequest, GenerateResponse
)
from api.inference import model_inference


# ─────────────────────────────────────────
#  Lifespan: runs at startup and shutdown
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model when server starts, clean up when it stops.
    
    Why here and not at module level?
    - Explicit control over when loading happens
    - Can handle errors gracefully
    - FastAPI's recommended pattern
    """
    print("🚀 Starting up...")
    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / "model" / "weights" / "tiny_gpt.pt"
    data_path = project_root / "data" / "tinyshakespeare.txt"
    
    if not weights_path.exists():
        print(f"⚠️  Weights not found at {weights_path}")
        print(f"   Run training first: cd model && python train.py")
    elif not data_path.exists():
        print(f"⚠️  Dataset not found at {data_path}")
    else:
        try:
            model_inference.load(str(weights_path), str(data_path))
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    
    yield  # Server is running, handling requests
    
    # Shutdown
    print("👋 Shutting down...")


# ─────────────────────────────────────────
#  Create the app
# ─────────────────────────────────────────
app = FastAPI(
    title="TinyGPT API",
    description="A GPT-style language model built from scratch, served over HTTP.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────
#  Endpoint 1: Health Check
# ─────────────────────────────────────────
@app.get("/health")
def health_check():
    """
    Simple health check.
    
    Used by:
    - Load balancers to know if this server is alive
    - Monitoring systems to track uptime
    - You, to verify the server started correctly
    """
    return {
        "status": "ok",
        "model_loaded": model_inference.model is not None,
    }


# ─────────────────────────────────────────
#  Endpoint 2: Echo
# ─────────────────────────────────────────
@app.post("/echo", response_model=EchoResponse)
def echo(request: EchoRequest):
    """
    Echo back the user's message.
    
    Purpose: Learn request/response flow without model complexity.
    The response_model parameter tells FastAPI:
      - Validate the OUTPUT matches EchoResponse shape
      - Generate accurate docs for what this endpoint returns
    """
    return EchoResponse(
        response=f"You said: {request.message}"
    )


# ─────────────────────────────────────────
#  Endpoint 3: Generate Text
# ─────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    """
    Generate text using the TinyGPT model.
    
    This is where Week 1 meets Week 2:
    - FastAPI receives the HTTP request
    - Pydantic validates the input
    - inference.py runs the model
    - The result comes back as JSON
    """
    # Check if model is loaded
    if model_inference.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Time the generation (you'll want this for monitoring later)
    start_time = time.time()
    
    try:
        generated = model_inference.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )
    
    elapsed = time.time() - start_time
    
    # Calculate tokens generated (total length minus prompt length)
    tokens_generated = len(generated) - len(request.prompt)
    
    # Log it (basic observability — you'll expand this in Phase 4)
    print(f"  📝 Generated {tokens_generated} tokens in {elapsed:.2f}s "
          f"| prompt: '{request.prompt[:30]}...'")
    
    return GenerateResponse(
        generated_text=generated,
        tokens_generated=tokens_generated,
    )