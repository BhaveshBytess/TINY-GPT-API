"""
The full RAG pipeline: retrieve → augment → generate.

This is the orchestration layer. It's an adapter too — the API will call
answer_question() and get back a clean result, never touching retrieval
or the LLM client directly.
"""
from typing import Dict, Optional
import logging

from rag.retriever import retriever
from rag.prompts import build_rag_prompt
from api.cloud_client import cloud_client, CloudAPIError  # Phase 1!

logger = logging.getLogger(__name__)


async def answer_question(
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    max_tokens: int = 500,
) -> Dict:
    """
    Answer a question using RAG.

    Returns a dict:
      {
        "answer": str,
        "sources": list of source metadata,
        "num_chunks_used": int,
        "grounded": bool   # False if no context was found
      }
    """
    # 1. Retrieve relevant chunks
    chunks = retriever.retrieve(
        query=question,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    # 2. Handle the no-context case — don't waste an LLM call
    if not chunks:
        logger.info("No relevant chunks for query: '%s'", question[:50])
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
            "num_chunks_used": 0,
            "grounded": False,
        }

    # 3. Build the augmented prompt
    prompt = build_rag_prompt(question, chunks)

    # 4. Call the LLM (your Phase 1 cloud_client)
    answer_text = await cloud_client.generate(prompt=prompt, max_tokens=max_tokens)

    # 5. Assemble the result with sources for citation
    sources = []
    seen = set()
    for chunk in chunks:
        src = chunk["metadata"].get("source", "unknown")
        if src not in seen:
            sources.append({
                "source": src,
                "score": round(chunk["score"], 3),
            })
            seen.add(src)

    logger.info(
        "RAG answer generated: %d chunks, %d sources",
        len(chunks), len(sources)
    )

    return {
        "answer": answer_text,
        "sources": sources,
        "num_chunks_used": len(chunks),
        "grounded": True,
    }