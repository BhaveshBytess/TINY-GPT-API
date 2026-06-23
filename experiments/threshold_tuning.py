"""
Explore how the score threshold affects what gets retrieved.
There's no universal correct value — it depends on your data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import embedding_model
from rag.retriever import Retriever


def main():
    embedding_model.load()
    retriever = Retriever(default_top_k=5)

    # A relevant query and an irrelevant query
    relevant = "How do I return a product?"
    irrelevant = "What's the weather like on Mars?"

    print("=" * 60)
    print("  THRESHOLD TUNING")
    print("=" * 60)

    for threshold in [0.0, 0.2, 0.3, 0.4, 0.5, 0.7]:
        rel = retriever.retrieve(relevant, score_threshold=threshold)
        irr = retriever.retrieve(irrelevant, score_threshold=threshold)
        print(f"\n  threshold = {threshold}")
        print(f"    relevant query   → {len(rel)} chunks pass")
        print(f"    irrelevant query → {len(irr)} chunks pass")

    print(f"\n  📌 Find the threshold where:")
    print(f"     • the RELEVANT query still returns chunks")
    print(f"     • the IRRELEVANT query returns ZERO")
    print(f"     That's your sweet spot for THIS knowledge base.")
    print(f"     (Different data needs different thresholds — always tune.)")


if __name__ == "__main__":
    main()