"""
Test retrieval quality against the ingested knowledge base.
Run AFTER scripts/ingest.py has populated the store.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import embedding_model
from rag.retriever import Retriever


def show(query, results):
    print(f"\n  QUERY: \"{query}\"")
    if not results:
        print("    (nothing passed the threshold — no relevant context)")
        return
    for r in results:
        src = r["metadata"].get("source", "?")
        print(f"    {r['score']:.3f}  [{src}]  {r['text'][:90]}...")


def main():
    embedding_model.load()
    retriever = Retriever(default_top_k=5, default_threshold=0.3)

    print("=" * 60)
    print("  TEST 1: Question answered in the docs")
    print("=" * 60)
    show("How long is the return window?",
         retriever.retrieve("How long is the return window?"))

    print("\n" + "=" * 60)
    print("  TEST 2: Question NOT in the docs (threshold should filter all)")
    print("=" * 60)
    show("What is the capital of France?",
         retriever.retrieve("What is the capital of France?"))

    print("\n" + "=" * 60)
    print("  TEST 3: Same query, different top_k")
    print("=" * 60)
    for k in [1, 3, 5]:
        results = retriever.retrieve("shipping times", top_k=k, score_threshold=0.0)
        print(f"\n  top_k={k}: returned {len(results)} chunks")
        for r in results:
            print(f"    {r['score']:.3f}  {r['text'][:70]}...")

    print("\n" + "=" * 60)
    print("  TEST 4: Metadata filter (restrict to one source)")
    print("=" * 60)
    # Adjust the source filename to match one of your documents
    show("anything",
         retriever.retrieve("policy", where={"source": "company_faq.txt"}))

    print("\n  Observe:")
    print("  - Test 1: relevant chunks, scores well above threshold")
    print("  - Test 2: empty - the threshold prevents hallucination fuel")
    print("  - Test 3: more k = more context but lower-relevance tail")
    print("  - Test 4: filter restricts search to a single document")


if __name__ == "__main__":
    main()
