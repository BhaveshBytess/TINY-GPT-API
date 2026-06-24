"""
Prove that the 'answer only from context' instruction prevents hallucination.
Compare RAG (grounded) vs raw LLM (ungrounded) on the same questions.
"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import embedding_model
from rag.retriever import retriever
from rag.prompts import build_rag_prompt
from api.cloud_client import cloud_client


async def main():
    embedding_model.load()

    # A question your docs DON'T answer but the LLM "knows" from training
    question = "Who is the CEO of Apple?"

    print("=" * 64)
    print(f"  QUESTION: {question}")
    print("  (Your docs are about a fictional company — they do NOT")
    print("   contain this answer. A grounded system should decline.)")
    print("=" * 64)

    # 1. RAW LLM (no grounding) — will happily answer from training data
    print("\n  --- RAW LLM (no RAG) ---")
    raw = await cloud_client.generate(prompt=question, max_tokens=100)
    print(f"  {raw}")

    # 2. RAG (grounded) — should decline because no relevant context
    print("\n  --- RAG (grounded) ---")
    chunks = retriever.retrieve(question, score_threshold=0.3)
    if not chunks:
        print("  (no chunks retrieved → RAG declines without calling LLM)")
    else:
        prompt = build_rag_prompt(question, chunks)
        grounded = await cloud_client.generate(prompt=prompt, max_tokens=100)
        print(f"  {grounded}")

    print("\n  📌 The raw LLM answers from training data (ungrounded).")
    print("     RAG declines or answers only from your docs (grounded).")
    print("     THIS is the difference RAG makes — controllable, traceable answers.")

    # 3. Now test the instruction's power directly:
    #    Give the LLM IRRELEVANT context and see if it resists answering
    print("\n" + "=" * 64)
    print("  INSTRUCTION STRESS TEST")
    print("  Give RAG-style prompt with WRONG context, see if it stays honest")
    print("=" * 64)
    fake_chunks = [{"text": "The sky is blue and grass is green.",
                    "score": 0.9, "metadata": {"source": "fake.txt"}}]
    prompt = build_rag_prompt(question, fake_chunks)
    result = await cloud_client.generate(prompt=prompt, max_tokens=100)
    print(f"  Q: {question}")
    print(f"  Context given: 'The sky is blue and grass is green.'")
    print(f"  A: {result}")
    print("\n  📌 A well-instructed LLM should say it can't answer from")
    print("     this context — NOT pull 'Tim Cook' from training data.")
    print("     If it answers anyway, your prompt needs strengthening.")


if __name__ == "__main__":
    asyncio.run(main())