"""
Explore where embeddings fail — these edge cases matter for RAG quality.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import EmbeddingModel


def main():
    model = EmbeddingModel()
    model.load()
    
    print("=" * 60)
    print("  EMBEDDING FAILURE CASES")
    print("=" * 60)
    
    # Failure 1: Negation
    print("\n  🔴 NEGATION — embeddings often miss 'not'")
    pairs = [
        ("I love this product", "I hate this product"),
        ("The test passed", "The test did not pass"),
        ("She is happy", "She is not happy"),
    ]
    for a, b in pairs:
        score = model.similarity(a, b)
        print(f"    {score:.3f}  \"{a}\" vs \"{b}\"")
    print("    ↑ These SHOULD be low (opposite meaning) but often aren't!")
    
    # Failure 2: Domain-specific jargon
    print("\n  🔴 DOMAIN JARGON — general models miss specialized terms")
    query = "What is the P/E ratio of AAPL?"
    candidates = [
        "Apple's price-to-earnings ratio is 28.5",
        "The price of apples at the store is $2.99",
        "AAPL stock performance this quarter",
    ]
    for c in candidates:
        score = model.similarity(query, c)
        print(f"    {score:.3f}  query: \"{query[:40]}\"")
        print(f"           match: \"{c[:50]}\"")
    
    # Failure 3: Length sensitivity
    print("\n  🔴 LENGTH SENSITIVITY — short vs long texts")
    short = "refund policy"
    long_text = ("Our comprehensive refund policy states that all purchases "
                "made within the last thirty days are eligible for a full "
                "refund provided that the original receipt is presented and "
                "the item is in its original packaging condition.")
    score = model.similarity(short, long_text)
    print(f"    {score:.3f}  short: \"{short}\"")
    print(f"           long:  \"{long_text[:60]}...\"")
    print("    ↑ Should be HIGH (same topic) — check if length hurts")
    
    # Failure 4: Number and fact sensitivity
    print("\n  🔴 NUMBERS — embeddings are weak on numerical facts")
    pairs = [
        ("The meeting is at 3 PM", "The meeting is at 5 PM"),
        ("The price is $10", "The price is $1000"),
        ("There are 5 items left", "There are 500 items left"),
    ]
    for a, b in pairs:
        score = model.similarity(a, b)
        print(f"    {score:.3f}  \"{a}\" vs \"{b}\"")
    print("    ↑ Very HIGH even though the facts are completely different!")
    print("    Embeddings capture topic, not specific facts.")
    
    print(f"\n{'=' * 60}")
    print("  📌 KEY TAKEAWAY:")
    print("  Embeddings capture TOPICAL similarity, not FACTUAL accuracy.")
    print("  They're great for finding relevant documents.")
    print("  They're terrible for verifying specific claims.")
    print("  This is why RAG needs the LLM — retrieval finds the context,")
    print("  the LLM extracts the precise answer from it.")
    print("=" * 60)


if __name__ == "__main__":
    main()