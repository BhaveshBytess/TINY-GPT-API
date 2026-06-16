"""
Compare chunking strategies on the same document.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.chunker import Chunker, chunk_document


SAMPLE = """The Refund Policy

Our refund policy allows customers to return products within 30 days of purchase. To be eligible for a refund, items must be in their original condition with all packaging intact. Refunds are processed back to the original payment method within 5 business days.

Shipping Information

Standard shipping takes 3 to 5 business days for domestic orders. International orders may take 10 to 15 business days and could incur additional customs fees. Premium members receive free express shipping on all orders.

Account Management

To reset your password, click the 'Forgot Password' link on the login page. We recommend enabling two-factor authentication for added security. You can update your email address and other details in the account settings page at any time."""


def show(name, chunks):
    print(f"\n{'=' * 60}")
    print(f"  {name}: {len(chunks)} chunks")
    print(f"{'=' * 60}")
    for i, c in enumerate(chunks[:3]):
        text = c["text"]
        print(f"  [chunk {i}] ({len(text)} chars)")
        print(f"    \"{text[:120]}{'...' if len(text) > 120 else ''}\"")
        print()


def main():
    doc = {"text": SAMPLE, "source": "faq.txt"}
    chunker = Chunker(chunk_size=200, overlap=30)

    show("FIXED-SIZE (200 chars, 30 overlap)",
         chunk_document(doc, chunker, "fixed"))
    show("SENTENCE-BASED",
         chunk_document(doc, chunker, "sentences"))
    show("RECURSIVE (production default)",
         chunk_document(doc, chunker, "recursive"))

    print("  📌 Observe:")
    print("  • Fixed-size splits mid-sentence, mid-word — crude")
    print("  • Sentence-based keeps sentences whole but may break topics")
    print("  • Recursive keeps paragraphs/topics together when possible")
    print("  • The 3 sections (refund/shipping/account) should ideally")
    print("    stay in separate chunks — which strategy does that best?")


if __name__ == "__main__":
    main()