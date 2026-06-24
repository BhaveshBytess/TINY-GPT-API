"""
Prompt templates for RAG.

The prompt is the #2 lever on RAG quality (after chunking). It controls
whether the LLM stays grounded in the retrieved context or drifts into
its training data.
"""
from typing import List, Dict


RAG_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the user's question based ONLY on "
    "the context provided below. If the context does not contain enough "
    "information to answer, say \"I don't have enough information to answer "
    "that.\" Do not use outside knowledge. When possible, cite the source "
    "of your answer."
)


def build_rag_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Build an augmented prompt from retrieved chunks.

    question: the user's question
    chunks:   list of {text, score, metadata} from the retriever

    Returns: the full prompt string to send to the LLM.
    """
    if not chunks:
        # No context — caller should ideally handle this before calling,
        # but we guard anyway.
        context_block = "(no relevant context found)"
    else:
        # Format each chunk with a source label for citation
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk["metadata"].get("source", "unknown")
            parts.append(f"[{i}] (source: {source})\n{chunk['text']}")
        context_block = "\n\n".join(parts)

    prompt = (
        f"{RAG_SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n"
        f"---\n"
        f"{context_block}\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt