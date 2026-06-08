"""
Explore vector store behavior at the edges.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


def main():
    embedder = EmbeddingModel()
    embedder.load()
    store = VectorStore(collection_name="limits_test")
    store.reset()

    docs = [
        "Python is a programming language.",
        "JavaScript runs in browsers.",
        "Machine learning uses neural networks.",
    ]
    embeddings = embedder.embed_batch(docs).tolist()
    store.add(
        ids=[str(i) for i in range(len(docs))],
        texts=docs,
        embeddings=embeddings,
    )

    # Limit 1: Query with NO relevant documents
    print("=" * 60)
    print("  LIMIT 1: Query unrelated to anything stored")
    print("=" * 60)
    q_vec = embedder.embed("What is the best recipe for chocolate cake?").tolist()
    results = store.query(q_vec, top_k=3)
    print("  Even with no relevant docs, it STILL returns the top-3.")
    print("  The scores tell you they're weak matches:")
    for r in results:
        print(f"    {r['score']:.3f}  {r['text']}")
    print("  📌 Lesson: vector DBs always return SOMETHING.")
    print("     You need a SCORE THRESHOLD to filter weak matches.")

    # Limit 2: Near-duplicate documents
    print(f"\n{'=' * 60}")
    print("  LIMIT 2: Near-duplicate documents")
    print("=" * 60)
    store2 = VectorStore(collection_name="dup_test")
    store2.reset()
    dup_docs = [
        "The cat sat on the mat.",
        "The cat sat on the mat.",        # exact duplicate
        "A cat was sitting on the mat.",  # near-duplicate
        "The weather is sunny today.",
    ]
    emb = embedder.embed_batch(dup_docs).tolist()
    store2.add(ids=[str(i) for i in range(len(dup_docs))], texts=dup_docs, embeddings=emb)
    q_vec = embedder.embed("Where is the cat?").tolist()
    results = store2.query(q_vec, top_k=4)
    print("  Query: 'Where is the cat?'")
    for r in results:
        print(f"    {r['score']:.3f}  {r['text']}")
    print("  📌 Lesson: duplicates eat up your top-k slots, crowding out")
    print("     diverse results. Production systems dedupe before storing.")

    # Limit 3: top_k larger than the collection
    print(f"\n{'=' * 60}")
    print("  LIMIT 3: Ask for more results than exist")
    print("=" * 60)
    q_vec = embedder.embed("programming").tolist()
    results = store.query(q_vec, top_k=100)  # only 3 docs exist
    print(f"  Asked for top_k=100, got {len(results)} (only 3 exist).")
    print("  📌 Lesson: it returns what it has, no error.")


if __name__ == "__main__":
    main()