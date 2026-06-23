"""
Runnable script to ingest documents into the RAG knowledge base.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --reset    # clear store first
"""
import sys, os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.logging_config import setup_logging  # reuse Phase 1 logging
from rag.embeddings import embedding_model
from rag.vector_store import vector_store
from rag.loader import load_directory
from rag.ingest import ingest_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="documents", help="documents folder")
    parser.add_argument("--reset", action="store_true", help="clear store first")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    setup_logging(level="INFO")

    # Load the embedding model
    embedding_model.load()

    # Optionally clear existing data
    if args.reset:
        vector_store.reset()
        print("Vector store reset.")

    # Load documents
    docs = load_directory(args.docs, pattern="*.txt")
    if not docs:
        print(f"No documents found in '{args.docs}/'. Add some .txt files.")
        return

    # Ingest
    summary = ingest_documents(
        docs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(f"\n{'=' * 50}")
    print(f"  Ingestion complete")
    print(f"  Documents: {summary['documents']}")
    print(f"  Chunks:    {summary['chunks']}")
    print(f"  Store now has {vector_store.count()} total chunks")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()