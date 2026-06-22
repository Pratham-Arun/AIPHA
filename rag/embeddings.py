"""
Embeddings Module
─────────────────
Creates an embedding model instance to convert text into vector
representations. Uses HuggingFace sentence-transformers.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFace embedding model instance.

    Returns:
        HuggingFaceEmbeddings configured with all-MiniLM-L6-v2.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return embeddings
