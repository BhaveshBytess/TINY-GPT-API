"""
Test the full RAG pipeline end-to-end.
Run AFTER ingesting documents (scripts/ingest.py).
"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import embedding_model
from rag.pipeline import answer_question


async def ask(question):
    print(f"\n{'=' * 64}")
    print(f"  Q: {question}")
    print(f"{'=' * 64}")
    result = await answer_question(question)
    print(f"  A: {result['answer']}")
    print(f"\n  grounded: {result['grounded']} | "
          f"chunks: {result['num_chunks_used']} | "
          f"sources: {[s['source'] for s in result['sources']]}")


async def main():
    embedding_model.load()

    # Questions answerable from your docs
    await ask("How long do I have to return a product?")
    await ask("Do premium members get free shipping?")
    await ask("How do I secure my account?")

    # Question NOT in your docs — should say "I don't know"
    await ask("What is the population of Tokyo?")


if __name__ == "__main__":
    asyncio.run(main())