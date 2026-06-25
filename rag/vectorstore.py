"""
Vector Store Module
───────────────────
Single interface to ChromaDB for storing and querying document embeddings.
"""

import gc
import shutil
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma

import config
from .embeddings import get_embedding_model


COLLECTION_NAME = "medical_documents"
CHROMA_SETTINGS = Settings(anonymized_telemetry=False, allow_reset=True)


class VectorStore:
    """Manages the ChromaDB lifecycle and vector operations."""

    def __init__(self, persist_directory: str | Path | None = None):
        self._persist_directory = Path(persist_directory or config.CHROMA_PERSIST_DIR)
        self._embedding_model = None
        self._chroma: Chroma | None = None

    def initialize(self) -> None:
        """Open a persistent client and ensure the collection is ready."""
        if self._chroma is not None:
            return

        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._embedding_model = get_embedding_model()
        self._chroma = Chroma(
            persist_directory=str(self._persist_directory),
            embedding_function=self._embedding_model,
            collection_name=COLLECTION_NAME,
            client_settings=CHROMA_SETTINGS,
        )

    def close(self) -> None:
        """Release the collection, client, and file handles."""
        if self._chroma is None:
            return

        chroma = self._chroma
        self._chroma = None
        self._embedding_model = None

        try:
            chroma._collection = None
        except Exception:
            pass

        try:
            client = chroma._client
            chroma._client = None
            del client
        except Exception:
            pass

        del chroma
        gc.collect()

    def delete(self) -> None:
        """Close the client and remove the persisted ChromaDB directory."""
        reset_done = False
        if self._chroma is not None:
            try:
                self._chroma._client.reset()
                reset_done = True
            except Exception:
                pass

        self.close()
        gc.collect()

        if self._persist_directory.exists() and not reset_done:
            shutil.rmtree(self._persist_directory)

        print(f"  Deleted vector store at: {self._persist_directory}")

    def create(self) -> None:
        """Create a fresh empty collection after teardown."""
        self.close()
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._embedding_model = get_embedding_model()
        self._chroma = Chroma(
            persist_directory=str(self._persist_directory),
            embedding_function=self._embedding_model,
            collection_name=COLLECTION_NAME,
            client_settings=CHROMA_SETTINGS,
        )

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        if self._chroma is None:
            return 0

        try:
            return self._chroma._collection.count()
        except Exception:
            return 0

    def exists(self, chunk_id: str) -> bool:
        """Return True if a chunk ID is already stored."""
        if self._chroma is None:
            return False

        result = self._chroma._collection.get(ids=[chunk_id])
        return bool(result and result.get("ids"))

    def add_documents(self, documents: list, ids: list[str]) -> int:
        """
        Insert documents that are not already present.

        Returns:
            Number of chunks successfully added.
        """
        if not documents or self._chroma is None:
            return 0

        result = self._chroma._collection.get(ids=ids)
        existing_ids = set(result.get("ids", []) if result else [])

        new_documents = []
        new_ids = []
        for document, chunk_id in zip(documents, ids):
            if chunk_id in existing_ids:
                continue
            new_documents.append(document)
            new_ids.append(chunk_id)

        if not new_documents:
            return 0

        self._chroma.add_documents(documents=new_documents, ids=new_ids)
        return len(new_ids)

    def search(
        self,
        query: str,
        k: int | None = None,
        filter_dict: dict | None = None,
    ) -> list:
        """Search the collection and return (Document, score) tuples."""
        if self._chroma is None:
            return []

        top_k = k or config.RETRIEVER_TOP_K
        return self._chroma.similarity_search_with_score(
            query,
            k=top_k,
            filter=filter_dict,
        )
