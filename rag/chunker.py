"""
Document chunking strategies.

Chunking is often the #1 lever on RAG quality — more impactful than
the embedding model or the LLM choice. This module makes the strategy
swappable so you can experiment.
"""
from typing import List
import re
import logging

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_fixed(self, text: str) -> List[str]:
        """
        Fixed-size chunks with overlap. Simplest strategy.
        Splits purely by character count, ignoring meaning.
        """
        if not text:
            return []
        chunks = []
        step = self.chunk_size - self.overlap
        for start in range(0, len(text), step):
            chunk = text[start:start + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def chunk_sentences(self, text: str) -> List[str]:
        """
        Group whole sentences into chunks under the size limit.
        Respects sentence boundaries — no mid-sentence splits.
        """
        # Simple sentence split (production uses nltk/spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) <= self.chunk_size:
                current += (" " + sentence if current else sentence)
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
        return chunks

    def chunk_recursive(self, text: str) -> List[str]:
        """
        Recursive splitting: paragraphs → sentences → fixed.
        The production-grade strategy. Keeps semantic units intact
        while respecting size limits.
        """
        # Level 1: split on paragraphs
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph keeps us under the limit, add it
            if len(current) + len(para) <= self.chunk_size:
                current += ("\n\n" + para if current else para)
            else:
                # Flush what we have
                if current:
                    chunks.append(current.strip())
                    current = ""

                # If the paragraph itself fits, start a new chunk with it
                if len(para) <= self.chunk_size:
                    current = para
                else:
                    # Level 2: paragraph too big, split on sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sentence in sentences:
                        if len(sentence) > self.chunk_size:
                            # Level 3: sentence too big, fall back to fixed
                            for i in range(0, len(sentence), self.chunk_size - self.overlap):
                                chunks.append(sentence[i:i + self.chunk_size].strip())
                        elif len(current) + len(sentence) <= self.chunk_size:
                            current += (" " + sentence if current else sentence)
                        else:
                            if current:
                                chunks.append(current.strip())
                            current = sentence

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if c.strip()]


def chunk_document(doc: dict, chunker: Chunker, strategy: str = "recursive") -> List[dict]:
    """
    Chunk a document dict into a list of chunk dicts with metadata.

    Each chunk carries source + index metadata for citation later.
    """
    text = doc["text"]
    if strategy == "fixed":
        chunks = chunker.chunk_fixed(text)
    elif strategy == "sentences":
        chunks = chunker.chunk_sentences(text)
    else:
        chunks = chunker.chunk_recursive(text)

    return [
        {
            "text": chunk,
            "metadata": {
                "source": doc.get("source", "unknown"),
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(chunks)
    ]