"""
Reset Vectorstore Script
────────────────────────
Safely closes and deletes the ChromaDB vector store, then recreates it.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.vectorstore import VectorStore


def reset_vectorstore():
    print("====================================")
    print("Reset Vector Store Script")
    print("====================================")

    vectorstore = VectorStore()
    vectorstore.initialize()

    print("  Closing vector store client...")
    vectorstore.close()

    print("  Deleting persistence directory...")
    vectorstore.delete()

    print("  Recreating empty collection...")
    vectorstore.create()

    print("  Vector store reset successfully.")


if __name__ == "__main__":
    reset_vectorstore()
