"""
RAG Pipeline Module
-------------------
Orchestrates the full RAG pipeline:
  1. Initialize vector store
  2. Index new documents from MongoDB incrementally
  3. Create retriever
  4. Retrieve relevant context for a query
"""

import gc
import time

import config
from database import DocumentService
from .indexer import Indexer
from .retriever import format_retrieved_docs, get_retriever
from .vectorstore import VectorStore


class RAGPipeline:
    """Manages the end-to-end RAG pipeline for the Health Assistant."""

    def __init__(self):
        self.vectorstore = VectorStore()
        self.retriever = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """
        Initialize the RAG pipeline.

        Opens ChromaDB, indexes pending MongoDB documents, and prepares retrieval.
        """
        print("\n-- Initializing RAG Pipeline --")

        self.vectorstore.initialize()

        indexer = Indexer(self.vectorstore)
        indexer.index_new_documents()

        self.retriever = get_retriever(self.vectorstore)
        self.is_initialized = True

        count = self.get_document_count()
        if count > 0:
            print(f"  RAG Pipeline ready! ({count} vector chunks loaded)\n")
            return True

        print("  RAG Pipeline ready, but no documents are currently indexed.")
        print("  Use the 'upload' command to add medical documents.\n")
        return False

    def index_new(self) -> int:
        """On-demand incremental indexing of pending documents."""
        indexer = Indexer(self.vectorstore)
        return indexer.index_new_documents()

    def retrieve(self, query: str, filter_dict: dict = None) -> tuple[str, list, int, int]:
        """
        Retrieve relevant medical context for a query.
        Applies relevance threshold to discard irrelevant chunks.

        Returns:
            Tuple of (formatted_context_string, relevant_results_list, discarded_count, search_time_ms)
        """
        if not self.is_initialized:
            return "No medical documents loaded.", [], 0, 0

        if self.get_document_count() == 0:
            return "No medical documents loaded in the database.", [], 0, 0

        start_time = time.time()
        all_results = self.vectorstore.search(query, filter_dict=filter_dict)
        search_time_ms = int((time.time() - start_time) * 1000)

        threshold = config.RETRIEVER_DISTANCE_THRESHOLD
        relevant_results = [(doc, score) for doc, score in all_results if score <= threshold]
        discarded_count = len(all_results) - len(relevant_results)

        return format_retrieved_docs(relevant_results), relevant_results, discarded_count, search_time_ms

    def rebuild(self) -> bool:
        """
        Force rebuild the entire vector store.

        Closes ChromaDB, deletes the persistence directory, resets MongoDB
        indexed status, and re-indexes every document from scratch.
        """
        print("\n-- Rebuilding RAG Pipeline --")

        print("  Step 1: Closing vector store...")
        self.retriever = None
        if self.vectorstore is not None:
            self.vectorstore.delete()
        self.vectorstore = None
        self.is_initialized = False
        gc.collect()

        print("  Step 2: Resetting MongoDB indexed status...")
        doc_service = DocumentService()
        reset_count = doc_service.reset_indexing_status()
        print(f"  Reset indexing status for {reset_count} document(s) in MongoDB.")

        print("  Step 3: Creating fresh collection and re-indexing...")
        self.vectorstore = VectorStore()
        self.vectorstore.create()
        indexer = Indexer(self.vectorstore)
        indexer.index_new_documents()
        self.retriever = get_retriever(self.vectorstore)
        self.is_initialized = True

        print("\n-- Rebuild Verification --")
        self._print_verify_report(doc_service)
        return self.get_document_count() > 0

    def verify_index(self) -> dict:
        """Run index health checks and return the verification summary."""
        doc_service = DocumentService()
        report = self._build_verify_report(doc_service)
        self._print_verify_report(doc_service, report)
        return report

    def shutdown(self) -> None:
        """Release retriever and vector store resources before application exit."""
        self.retriever = None
        if self.vectorstore is not None:
            self.vectorstore.close()
            self.vectorstore = None
        self.is_initialized = False
        gc.collect()

    def get_document_count(self) -> int:
        """Return the number of vectors in the store."""
        if self.vectorstore is None:
            return 0
        return self.vectorstore.count()

    def _duplicate_count(self) -> int:
        """Count duplicate chunk IDs currently stored in ChromaDB."""
        if self.vectorstore is None or self.vectorstore._chroma is None:
            return 0

        try:
            result = self.vectorstore._chroma._collection.get()
            ids = result.get("ids", []) if result else []
            return len(ids) - len(set(ids))
        except Exception:
            return 0

    def _build_verify_report(self, doc_service: DocumentService) -> dict:
        all_docs = doc_service.list_documents()
        total_docs = len(all_docs)
        indexed_docs = sum(1 for doc in all_docs if doc.get("indexed"))
        pending_docs = total_docs - indexed_docs
        expected_chunks = sum(doc.get("chunk_count", 0) for doc in all_docs)
        ghost_docs = sum(
            1 for doc in all_docs if doc.get("indexed") and doc.get("chunk_count", 0) == 0
        )
        vector_count = self.get_document_count()
        duplicate_chunks = self._duplicate_count()

        healthy = (
            pending_docs == 0
            and indexed_docs == total_docs
            and ghost_docs == 0
            and duplicate_chunks == 0
            and vector_count == expected_chunks
            and vector_count > 0
        )

        if healthy:
            status = "Healthy"
        elif ghost_docs > 0:
            status = f"WARNING - {ghost_docs} documents marked indexed without chunks. Run 'rebuild'."
        elif pending_docs > 0:
            status = f"{pending_docs} document(s) not yet indexed"
        elif duplicate_chunks > 0:
            status = f"WARNING - {duplicate_chunks} duplicate chunks found."
        elif vector_count != expected_chunks:
            status = (
                f"WARNING - Mismatch between MongoDB expected chunks ({expected_chunks}) "
                f"and ChromaDB ({vector_count})"
            )
        else:
            status = "WARNING - No vectors in ChromaDB"

        return {
            "total_docs": total_docs,
            "indexed_docs": indexed_docs,
            "pending_docs": pending_docs,
            "expected_chunks": expected_chunks,
            "ghost_docs": ghost_docs,
            "vector_count": vector_count,
            "duplicate_chunks": duplicate_chunks,
            "status": status,
            "healthy": healthy,
        }

    def _print_verify_report(self, doc_service: DocumentService, report: dict | None = None) -> None:
        report = report or self._build_verify_report(doc_service)

        print(f"  MongoDB Documents : {report['total_docs']}")
        print(f"  Indexed           : {report['indexed_docs']}")
        if report["ghost_docs"] > 0:
            print(f"  Ghost Documents   : {report['ghost_docs']}")
        print(f"  Expected Chunks   : {report['expected_chunks']}")
        print(f"  Vector Count      : {report['vector_count']}")
        print(f"  Duplicates        : {report['duplicate_chunks']}")
        print(f"  Status            : {report['status']}")
