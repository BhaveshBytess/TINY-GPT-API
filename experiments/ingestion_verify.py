"""
Verify the ingestion pipeline worked: query for known content.
Run AFTER scripts/ingest.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import embedding_model
from rag.vector_store import vector_store


def main():
    embedding_model.load()

    print(f"Store contains {vector_store.count()} chunks\n")

    queries = [
        "How long do I have to return something?",
        "Do premium members pay for shipping?",
        "How do I make my account more secure?",
        "What is the meaning of life?",   # should score LOW — not in docs
    ]

    for q in queries:
        print(f"{'=' * 60}")
        print(f"  QUERY: \"{q}\"")
        print(f"{'=' * 60}")
        q_vec = embedding_model.embed(q).tolist()
        results = vector_store.query(q_vec, top_k=2)
        for r in results:
            src = r["metadata"].get("source", "?")
            print(f"  {r['score']:.3f}  [{src}]")
            print(f"    \"{r['text'][:120]}...\"")
        print()

    print("  NOTE: The first 3 queries should return relevant chunks with")
    print("     good scores. The last one should score LOW — proof that")
    print("     your store doesn't hallucinate relevance for unknown topics.")


if __name__ == "__main__":
    main()