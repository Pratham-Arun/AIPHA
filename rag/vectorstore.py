"""
Vector Store Module
───────────────────
Manages ChromaDB vector database for storing and querying document embeddings.
Supports creating a new store from documents and loading an existing one.
"""

from pathlib import Path
from langchain_chroma import Chroma
from .embeddings import get_embedding_model
import config


COLLECTION_NAME = "medical_documents"


def create_vectorstore(chunks: list, persist_directory: str | Path = None) -> Chroma:
    """
    Create a new ChromaDB vector store from document chunks.

    Args:
        chunks:            List of Document objects (chunks) to embed and store.
        persist_directory: Directory to persist the database
                           (default from config).

    Returns:
        Chroma vector store instance.
    """
    persist_directory = str(persist_directory or config.CHROMA_PERSIST_DIR)

    embedding_model = get_embedding_model()

    print(f"  Creating vector store with {len(chunks)} chunk(s)...")
    print(f"  Persist directory: {persist_directory}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
    )

    print(f"  Vector store created successfully!")
    return vectorstore


def load_vectorstore(persist_directory: str | Path = None) -> Chroma | None:
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        persist_directory: Directory where the database is persisted
                           (default from config).

    Returns:
        Chroma vector store instance, or None if not found.
    """
    persist_directory = str(persist_directory or config.CHROMA_PERSIST_DIR)

    if not Path(persist_directory).exists():
        return None

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )

    # Check if the collection actually has documents
    count = vectorstore._collection.count()
    if count == 0:
        return None

    print(f"  Loaded existing vector store: {count} vector(s)")
    return vectorstore


def get_or_create_vectorstore(persist_directory: str | Path = None) -> Chroma:
    """
    Get the existing vector store or create an empty one if it doesn't exist.
    Used for incremental indexing.
    """
    persist_directory = str(persist_directory or config.CHROMA_PERSIST_DIR)
    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    
    return vectorstore


def delete_vectorstore(persist_directory: str | Path = None) -> None:
    """
    Delete the persisted vector store directory.

    Args:
        persist_directory: Directory to delete (default from config).
    """
    import shutil

    persist_directory = Path(persist_directory or config.CHROMA_PERSIST_DIR)
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
        print(f"  Deleted vector store at: {persist_directory}")
    else:
        print(f"  No vector store found at: {persist_directory}")
