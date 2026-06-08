"""
Explore embedding similarity across different sentence types.
This is the experiment that builds intuition for what embeddings capture.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import EmbeddingModel
import numpy as np


def print_similarity(label, pairs, model):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    for text_a, text_b in pairs:
        score = model.similarity(text_a, text_b)
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"  {score:.3f}  {bar}")
        print(f"         A: \"{text_a[:50]}\"")
        print(f"         B: \"{text_b[:50]}\"")
        print()


def main():
    model = EmbeddingModel()
    model.load()
    
    similar_pairs = [
        ("The cat sat on the mat", "A kitten rested on a rug"),
        ("How do I return a product?", "What is the refund policy?"),
        ("The stock market crashed today", "Financial markets saw steep declines"),
    ]
    
    different_pairs = [
        ("The cat sat on the mat", "Quantum computing uses qubits"),
        ("How do I return a product?", "The weather is nice today"),
        ("Python is a programming language", "A python is a large snake"),
    ]
    
    tricky_pairs = [
        ("I went to the bank to deposit money", "I sat on the river bank"),
        ("The bat flew across the cave", "He swung the bat at the ball"),
        ("Apple released a new iPhone", "I ate an apple for breakfast"),
    ]
    
    paraphrase_pairs = [
        ("The movie was great", "I really enjoyed the film"),
        ("She is not happy", "She feels sad"),
        ("The car is fast", "The vehicle has high speed"),
    ]
    
    print("=" * 60)
    print("  EMBEDDING SIMILARITY EXPLORER")
    print("  Model: all-MiniLM-L6-v2 (384 dimensions)")
    print("=" * 60)
    
    print_similarity("Similar meaning (expect HIGH: 0.7+)", similar_pairs, model)
    print_similarity("Different meaning (expect LOW: 0.0-0.3)", different_pairs, model)
    print_similarity("Tricky: same words, different meaning", tricky_pairs, model)
    print_similarity("Paraphrases: different words, same meaning", paraphrase_pairs, model)
    
    print("=" * 60)
    print("  📌 What to observe:")
    print("  • Similar pairs should score 0.5-0.9")
    print("  • Different pairs should score 0.0-0.3")
    print("  • Tricky pairs reveal embedding limitations")
    print("  • Paraphrases should score HIGH despite different words")
    print("    (this is what makes semantic search possible)")
    print("=" * 60)


if __name__ == "__main__":
    main()