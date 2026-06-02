# TinyGPT API

A GPT-style language model built **from scratch** in PyTorch, served through a
production-structured FastAPI service with cloud LLM fallback.

No HuggingFace. No pre-built transformer modules. Every component — scaled
dot-product attention, multi-head attention, transformer blocks, training loop,
and text generation — is implemented from first principles. Seven ablation
experiments demonstrate what each component contributes by removing it and
observing what breaks.

---

## Why This Project Exists

Most "I learned transformers" projects stop at a working model copied from a
tutorial. This one goes two steps further:

1. **Understanding through failure.** Each architectural component was removed
   in a controlled experiment to observe its specific failure mode — proving
   *why* it exists, not just *that* it exists.
2. **From model to service.** The trained model is wrapped in a real API with
   the production patterns an AI service actually needs: input validation,
   structured logging, request tracing, global error handling, and cloud
   model fallback.

---

## What's Inside

### The Model (from scratch)

- **Scaled dot-product attention** with causal masking
- **Multi-head attention** — 4 heads, with the split/concat dimension handling
- **Pre-norm transformer blocks** — LayerNorm → MHA → residual → LayerNorm → FFN → residual
- **TinyGPT** — token + positional embeddings, N stacked blocks, final norm, linear head
- **Character-level training** on TinyShakespeare (~1MB)
- **Text generation** with greedy, temperature, and top-k sampling

### Model Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| num_heads | 4 |
| num_layers | 4 |
| d_ff | 512 |
| max_seq_len | 256 |
| vocab_size | 65 (character-level) |
| dropout | 0.1 |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01) |
| Training | 5000 steps, batch size 64 |

### The Service

- **FastAPI** with four endpoints: `/health`, `/echo`, `/generate`, `/chat`
- **Model-selector routing** — `/chat` routes between local TinyGPT and a cloud LLM (Mistral)
- **Pydantic validation** — typed request/response schemas with field constraints
- **Structured logging** — timestamps, severity levels, module names
- **Request tracing** — unique request ID per request, propagated through logs and response headers
- **Global exception handling** — clean error responses, no stack traces leaked to users
- **CORS support** — browser frontends can call the API

---

## What I Learned (Ablation Experiments)

> Replace the bracketed notes with your actual observed numbers from each run.

### Removing √dₖ scaling
As d_k grew from 8 to 1024, the unscaled attention's entropy collapsed toward
zero — softmax became a near one-hot distribution, putting almost all weight
on a single token. The scaled version stayed stable. This is why scaling exists:
without it, large dot products saturate softmax and gradients vanish.
*(Your numbers: entropy went from [X] to [Y] unscaled vs stable [Z] scaled.)*

### Removing the causal mask
Without masking, position 0 distributed attention across all future positions —
information leakage. In training this lets the model "cheat" by seeing the
answer instead of predicting it, producing a strategy that fails at inference
where future tokens don't exist.

### 1 head vs 4 vs 8 heads
More heads converged faster / to lower loss because each head can specialize in
a different relationship type. A single head is forced to compress all patterns
into one attention distribution.
*(Your numbers: 1-head final loss [X] vs 8-head [Y].)*

### Removing residual connections
At 1 layer the difference was small; at 8 layers the no-residual model's
first-layer gradient norms collapsed toward zero and loss stalled. The residual
connection guarantees a gradient of at least 1 reaching early layers.
*(Your numbers: no-residual loss stalled at [X] vs [Y] with residuals.)*

### Removing LayerNorm
Training became unstable — loss spiked or hit NaN within a few hundred steps,
worse at greater depth. LayerNorm keeps activations in a stable range.

### Removing positional embeddings
The model still learned character frequencies (loss dropped), but generated text
lost all word order and grammatical structure — proving attention is
permutation-invariant and needs explicit position information.

### Depth comparison (1 vs 4 vs 8 layers)
Deeper models produced more coherent text and captured longer-range dependencies,
at the cost of more parameters and slower training.

---

## Architecture

```
tiny-gpt-api/
├── model/                      # The model (Week 1)
│   ├── attention.py            # scaled_dot_product_attention, MultiHeadAttention
│   ├── transformer.py          # FeedForward, TransformerBlock, TinyGPT
│   ├── dataset.py              # CharDataset + TinyShakespeare loader (shared vocab)
│   ├── train.py                # training loop, evaluation, generation
│   ├── data/tinyshakespeare.txt
│   └── weights/tiny_gpt.pt
│
├── api/                        # The service (Week 2)
│   ├── main.py                 # FastAPI app + endpoints
│   ├── schemas.py              # Pydantic request/response models
│   ├── inference.py            # Model adapter (strings in, strings out)
│   ├── cloud_client.py         # Async Mistral API adapter
│   ├── config.py               # Settings from .env
│   ├── logging_config.py       # Structured logging setup
│   ├── middleware.py           # Request ID + lifecycle logging
│   └── exceptions.py           # Global exception handlers
│
├── experiments/                # 7 ablation experiments
├── .env                        # API keys (gitignored)
├── requirements.txt
└── README.md
```

**Key design decision — the adapter boundary:** `inference.py` and
`cloud_client.py` are adapters. The API layer never touches PyTorch tensors;
the model never sees an HTTP request. This decoupling is why adding the cloud
provider required zero changes to the endpoint code.

---

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Set a cloud API key for the /chat cloud route
echo "MISTRAL_API_KEY=your-key-here" > .env

# Train the model (auto-downloads TinyShakespeare if missing)
cd model && python train.py && cd ..

# Run the API
uvicorn api.main:app --reload --port 8000

# Open the interactive docs
# http://localhost:8000/docs
```

---

## API Reference

| Method | Path | Body | Description |
|--------|------|------|-------------|
| GET | `/health` | — | Status + whether model/cloud are ready |
| POST | `/echo` | `{"message": "..."}` | Echoes the message (learning endpoint) |
| POST | `/generate` | `{"prompt": "...", "max_tokens": 50}` | Generate with local TinyGPT |
| POST | `/chat` | `{"message": "...", "model": "tiny\|cloud", "max_tokens": 200}` | Routes between local and cloud |

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain attention in one sentence", "model": "cloud"}'
```

---

## Sample Output

> Paste your real generated text here at different training stages.

```
Step 500  (loss ~2.2): [early garbage with some real letters]
Step 2000 (loss ~1.5): [real words emerging]
Step 5000 (loss ~1.2): [semi-coherent Shakespeare-ish text]
```

---

## What I'd Add for Production

This is a learning project. To make it production-grade, the next steps would be:

- **Authentication** — API keys or OAuth to control access
- **Rate limiting** — per-client request quotas
- **Request batching** — combine concurrent requests into a single GPU forward pass (vLLM/TGI-style) instead of `--workers` duplicating the model
- **Metrics & tracing** — Prometheus for metrics, OpenTelemetry for distributed traces, on top of the existing structured logging
- **Token streaming** — Server-Sent Events to stream tokens as generated
- **Containerization & deployment** — Docker + an orchestrator, with the model weights in object storage
- **Caching** — cache identical prompts to save compute

---

## Tech Stack

Python · PyTorch · FastAPI · Pydantic · httpx · Uvicorn · Mistral API

---

*Built as Phase 1 of an AI Engineer learning roadmap — the foundation for
later work on RAG, agentic systems, and MLOps.*