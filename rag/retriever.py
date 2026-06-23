"""
Retriever: the query-time counterpart to the ingestion pipeline.

Takes a question, returns the most relevant chunks from the knowledge base.
Adapter pattern: the RAG pipeline calls retrieve(), never touches the
embedding model or vector store directly.
"""
from typing import List, Dict, Optional
import logging

from rag.embeddings import embedding_model
from rag.vector_store import vector_store

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, default_top_k: int = 5,
                 default_threshold: float = 0.3):
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query.

        query:           the user's question
        top_k:           how many chunks to consider (before threshold)
        score_threshold: drop chunks below this similarity score
        where:           optional metadata filter, e.g. {"source": "policy.pdf"}

        Returns: list of {id, text, score, metadata}, possibly empty.
        """
        top_k = top_k if top_k is not None else self.default_top_k
        threshold = score_threshold if score_threshold is not None else self.default_threshold

        # 1. Embed the query — SAME model used at ingestion
        query_vector = embedding_model.embed(query).tolist()

        # 2. Search the vector store
        raw_results = vector_store.query(
            query_embedding=query_vector,
            top_k=top_k,
            where=where,
        )

        # 3. Apply the score threshold
        filtered = [r for r in raw_results if r["score"] >= threshold]

        logger.info(
            "Retrieval: query='%s' | %d raw → %d above threshold %.2f",
            query[:50], len(raw_results), len(filtered), threshold
        )

        return filtered


# Singleton
retriever = Retriever()