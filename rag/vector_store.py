"""
Vector store wrapper around ChromaDB.

Adapter pattern: the RAG pipeline calls add()/query(), never touches
ChromaDB's API directly. Swapping to Pinecone or Qdrant later means
changing this file only.
"""
import chromadb
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Stores document chunks as vectors and searches them by similarity.
    """

    def __init__(self, persist_path: str = "./chroma_db",
                 collection_name: str = "documents"):
        self.persist_path = persist_path
        self.collection_name = collection_name
        # PersistentClient saves to disk; data survives restarts
        self.client = chromadb.PersistentClient(path=persist_path)
        # cosine space matches our normalized embeddings from 3.1
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Vector store ready: %s (%d documents)",
            collection_name, self.collection.count()
        )

    def add(self, ids: List[str], texts: List[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the store.

        ids:        unique string IDs for each chunk
        texts:      the original text of each chunk
        embeddings: the vector for each chunk (list of floats)
        metadatas:  optional dict of structured fields per chunk
        """
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Added %d documents. Total: %d", len(ids), self.collection.count())

    def query(self, query_embedding: List[float], top_k: int = 5,
              where: Optional[Dict] = None) -> List[Dict]:
        """
        Find the top_k most similar documents.

        Returns a list of dicts with: id, text, score, metadata.
        'where' is an optional metadata filter, e.g. {"category": "refunds"}
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        # ChromaDB returns parallel lists; flatten into clean dicts
        output = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        distances = results["distances"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)

        for i in range(len(ids)):
            output.append({
                "id": ids[i],
                "text": docs[i],
                # cosine distance → similarity: similarity = 1 - distance
                "score": 1.0 - distances[i],
                "metadata": metas[i] or {},
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete the collection and recreate it empty."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store reset.")


# Singleton
vector_store = VectorStore()