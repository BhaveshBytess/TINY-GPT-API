"""
Ingestion pipeline: load → chunk → embed → store.

This is the offline step that builds the knowledge base. Run it when
documents are added or changed. The query side (retrieval) reads from
what this produces.
"""
from typing import List, Dict
import logging

from rag.embeddings import embedding_model
from rag.vector_store import vector_store
from rag.chunker import Chunker, chunk_document

logger = logging.getLogger(__name__)


def ingest_documents(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50,
    strategy: str = "recursive",
) -> Dict:
    """
    Ingest a list of documents into the vector store.

    documents: list of {"text": ..., "source": ...} dicts (from the loader)

    Returns a summary dict: {documents, chunks}.
    """
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)

    all_ids = []
    all_texts = []
    all_metadatas = []

    # Phase 1: chunk every document
    for doc in documents:
        chunks = chunk_document(doc, chunker, strategy=strategy)
        source = doc.get("source", "unknown")

        for chunk in chunks:
            idx = chunk["metadata"]["chunk_index"]
            # Stable, unique id: re-ingesting updates in place
            chunk_id = f"{source}_{idx}"
            all_ids.append(chunk_id)
            all_texts.append(chunk["text"])
            all_metadatas.append(chunk["metadata"])

        logger.info("Chunked %s into %d chunks", source, len(chunks))

    if not all_texts:
        logger.warning("No chunks produced — nothing to ingest.")
        return {"documents": len(documents), "chunks": 0}

    # Phase 2: embed all chunks in one batch (fast)
    logger.info("Embedding %d chunks...", len(all_texts))
    embeddings = embedding_model.embed_batch(all_texts).tolist()

    # Phase 3: store everything
    vector_store.add(
        ids=all_ids,
        texts=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    summary = {"documents": len(documents), "chunks": len(all_texts)}
    logger.info("Ingestion complete: %s", summary)
    return summary