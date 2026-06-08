"""
Demo: store documents in the vector store and search them semantically.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


def main():
    # Fresh store for the demo
    embedder = EmbeddingModel()
    embedder.load()

    store = VectorStore(collection_name="demo")
    store.reset()  # start clean

    documents = [
        {"id": "1", "text": "Our refund policy allows returns within 30 days.", "category": "policy"},
        {"id": "2", "text": "Shipping takes 3-5 business days for domestic orders.", "category": "shipping"},
        {"id": "3", "text": "To reset your password, click 'Forgot Password' on login.", "category": "account"},
        {"id": "4", "text": "Premium members get free express shipping.", "category": "shipping"},
        {"id": "5", "text": "Refunds are processed back to the original payment method.", "category": "policy"},
        {"id": "6", "text": "Two-factor authentication adds security to your account.", "category": "account"},
        {"id": "7", "text": "International shipping may incur customs fees.", "category": "shipping"},
        {"id": "8", "text": "You can update your email in account settings.", "category": "account"},
    ]

    # Embed and store
    texts = [d["text"] for d in documents]
    embeddings = embedder.embed_batch(texts).tolist()
    store.add(
        ids=[d["id"] for d in documents],
        texts=texts,
        embeddings=embeddings,
        metadatas=[{"category": d["category"]} for d in documents],
    )

    # Run some queries
    queries = [
        "How do I get my money back?",
        "When will my package arrive?",
        "I forgot my login credentials",
    ]

    for q in queries:
        print(f"\n{'=' * 60}")
        print(f"  QUERY: \"{q}\"")
        print(f"{'=' * 60}")
        q_vec = embedder.embed(q).tolist()
        results = store.query(q_vec, top_k=3)
        for r in results:
            print(f"  {r['score']:.3f}  [{r['metadata'].get('category')}]  {r['text']}")

    # Demonstrate metadata filtering
    print(f"\n{'=' * 60}")
    print(f"  QUERY with filter: \"shipping question\" WHERE category='shipping'")
    print(f"{'=' * 60}")
    q_vec = embedder.embed("shipping question").tolist()
    results = store.query(q_vec, top_k=5, where={"category": "shipping"})
    for r in results:
        print(f"  {r['score']:.3f}  [{r['metadata'].get('category')}]  {r['text']}")

    print(f"\n  📌 Notice: 'money back' query returns REFUND docs")
    print(f"     even though they don't contain those words. Semantic search!")
    print(f"     And the filter restricts results to only shipping docs.")


if __name__ == "__main__":
    main()