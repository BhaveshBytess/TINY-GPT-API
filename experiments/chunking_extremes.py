"""
Show how chunk size affects retrieval quality.
Tiny chunks lose context; huge chunks dilute relevance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.chunker import Chunker, chunk_document


SAMPLE = """Our refund policy allows customers to return products within 30 days of purchase. To be eligible for a refund, items must be in their original condition with all packaging intact. Refunds are processed back to the original payment method within 5 business days. For defective items, we offer free return shipping and expedited refunds. Store credit is available as an alternative to a cash refund and never expires. International customers should note that original shipping costs are non-refundable."""


def test_chunk_size(embedder, chunk_size, overlap, query):
    store = VectorStore(collection_name=f"test_{chunk_size}")
    store.reset()

    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunk_document({"text": SAMPLE, "source": "policy.txt"},
                           chunker, "recursive")

    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_batch(texts).tolist()
    store.add(
        ids=[str(i) for i in range(len(texts))],
        texts=texts,
        embeddings=embeddings,
    )

    q_vec = embedder.embed(query).tolist()
    results = store.query(q_vec, top_k=1)

    print(f"\n  chunk_size={chunk_size:>4d}  →  {len(chunks)} chunks")
    print(f"  top result (score {results[0]['score']:.3f}):")
    print(f"    \"{results[0]['text'][:150]}\"")


def main():
    embedder = EmbeddingModel()
    embedder.load()

    query = "Can I get store credit instead of a refund?"

    print("=" * 60)
    print(f"  QUERY: \"{query}\"")
    print(f"  The answer is in the sentence about store credit.")
    print("=" * 60)

    # Too small — fragments lose context
    test_chunk_size(embedder, 50, 10, query)
    # Just right — coherent units
    test_chunk_size(embedder, 200, 30, query)
    # Too big — whole doc in one chunk, diluted
    test_chunk_size(embedder, 2000, 0, query)

    print(f"\n  📌 Observe:")
    print(f"  • size=50: the store-credit info is fragmented across tiny chunks")
    print(f"  • size=200: clean chunk with the store-credit sentence + context")
    print(f"  • size=2000: one giant chunk — retrieval 'works' but you hand")
    print(f"    the LLM the ENTIRE document instead of the relevant part")


if __name__ == "__main__":
    main()