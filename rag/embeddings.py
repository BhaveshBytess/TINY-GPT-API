"""
Embedding model wrapper.

Same adapter pattern as inference.py — the RAG pipeline calls embed(),
never touches the model directly. Swapping to a different embedding model
(OpenAI, Mistral, Cohere) means changing this file only.
"""
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wraps a sentence-transformers model for text embedding.
    
    Interface:
      embed("some text") → numpy array of shape (384,)
      embed_batch(["text1", "text2"]) → numpy array of shape (N, 384)
    """
    
    def __init__(self):
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"
        self.dimension = 384  # This model outputs 384-dim vectors
    
    def load(self):
        """Load the embedding model. Called once at startup."""
        from sentence_transformers import SentenceTransformer
        
        logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        logger.info(
            "Embedding model loaded. Dimension: %d", self.dimension
        )
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Returns: numpy array of shape (dimension,)
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load() first.")
        
        # normalize_embeddings=True makes cosine similarity = dot product
        # (vectors are unit length, so cosine = dot product)
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts efficiently in one batch.
        
        Returns: numpy array of shape (len(texts), dimension)
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load() first.")
        
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vectors
    
    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Returns: float between -1 and 1 (typically 0 to 1 for text)
        """
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        # With normalized vectors, cosine similarity = dot product
        return float(np.dot(vec_a, vec_b))


# Singleton
embedding_model = EmbeddingModel()