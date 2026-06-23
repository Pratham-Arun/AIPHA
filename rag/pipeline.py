"""
RAG Pipeline Module
-------------------
Orchestrates the full RAG pipeline:
  1. Load or create vector store
  2. Index new documents from MongoDB incrementally
  3. Create retriever
  4. Retrieve relevant context for a query
"""

from .vectorstore import load_vectorstore, delete_vectorstore, get_or_create_vectorstore, close_vectorstore
from .retriever import get_retriever, format_retrieved_docs
from .indexer import Indexer
from database import DocumentService
import config
import gc


class RAGPipeline:
    """
    Manages the end-to-end RAG pipeline for the Health Assistant.
    """

    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """
        Initialize the RAG pipeline.
        
        Attempts to load an existing vector store, then runs the incremental
        indexer to pick up any new documents from MongoDB.
        """
        print("\n-- Initializing RAG Pipeline --")

        # 1. Load or Create Vector Store
        self.vectorstore = get_or_create_vectorstore()
        
        # 2. Run Indexer for new documents
        indexer = Indexer()
        indexer.index_new_documents()
        
        # 3. Create Retriever
        # Even if collection is empty, the retriever can be created
        self.retriever = get_retriever(self.vectorstore)
        self.is_initialized = True
        
        count = self.get_document_count()
        if count > 0:
            print(f"  RAG Pipeline ready! ({count} vector chunks loaded)\n")
            return True
        else:
            print("  RAG Pipeline ready, but no documents are currently indexed.")
            print("  Use the 'upload' command to add medical documents.\n")
            return False

    def index_new(self) -> int:
        """
        On-demand incremental indexing of pending documents.
        """
        indexer = Indexer()
        return indexer.index_new_documents()

    def retrieve(self, query: str, filter_dict: dict = None) -> tuple[str, list, int, int]:
        """
        Retrieve relevant medical context for a query.
        Applies relevance threshold to discard irrelevant chunks.
        
        Returns:
            Tuple of (formatted_context_string, relevant_results_list, discarded_count, search_time_ms)
        """
        import time
        if not self.is_initialized or self.vectorstore is None:
            return "No medical documents loaded.", [], 0, 0

        # If vectorstore is empty, retriever will throw an error or return nothing
        if self.get_document_count() == 0:
            return "No medical documents loaded in the database.", [], 0, 0

        start_time = time.time()
        from .retriever import search_with_scores, format_retrieved_docs
        all_results = search_with_scores(self.vectorstore, query, filter_dict=filter_dict)
        search_time_ms = int((time.time() - start_time) * 1000)
        
        # Apply relevance threshold - discard chunks with distance > threshold
        threshold = config.RETRIEVER_DISTANCE_THRESHOLD
        relevant_results = [(doc, score) for doc, score in all_results if score <= threshold]
        discarded_count = len(all_results) - len(relevant_results)
        
        return format_retrieved_docs(relevant_results), relevant_results, discarded_count, search_time_ms

    def rebuild(self) -> bool:
        """
        Force rebuild the entire vector store.

        Releases all existing ChromaDB/PersistentClient references so Windows
        can delete the persistence directory (avoiding WinError 32), then resets
        MongoDB indexed status and re-indexes every document from scratch.
        """
        print("\n-- Rebuilding RAG Pipeline --")

        # ── Step 1: Release all references to the vectorstore/retriever ──
        # This is critical on Windows: the ChromaDB PersistentClient holds
        # open SQLite file handles. We must close them before rmtree().
        print("  Step 1: Releasing ChromaDB file handles...")
        close_vectorstore(self.vectorstore)   # closes PersistentClient + SQLite
        self.retriever = None
        self.vectorstore = None
        self.is_initialized = False
        gc.collect()   # ensure CPython reference-counts drop to zero now

        # ── Step 2: Delete the persistence directory ──
        print("  Step 2: Deleting ChromaDB persistence directory...")
        try:
            delete_vectorstore()
        except PermissionError:
            # One last-resort attempt: wait a moment longer and retry once
            import time
            print("  Retrying deletion after short wait...")
            time.sleep(1.0)
            gc.collect()
            delete_vectorstore()

        # ── Step 3: Reset MongoDB indexed status ──
        print("  Step 3: Resetting MongoDB indexed status...")
        doc_service = DocumentService()
        reset_count = doc_service.reset_indexing_status()
        print(f"  Reset indexing status for {reset_count} document(s) in MongoDB.")

        # ── Step 4: Create a fresh vector store and run the indexer ──
        print("  Step 4: Re-indexing all documents...")
        self.vectorstore = get_or_create_vectorstore()
        indexer = Indexer()
        indexer.index_new_documents()
        self.retriever = get_retriever(self.vectorstore)
        self.is_initialized = True

        # ── Step 5: Verification ──
        count = self.get_document_count()
        total_docs = len(doc_service.list_documents())
        print("\n-- Rebuild Verification --")
        print(f"  MongoDB Documents : {total_docs}")
        print(f"  Vector Chunks     : {count}")
        if count > 0:
            print("  Status            : PASSED")
            print(f"  RAG Pipeline ready! ({count} vector chunks loaded)")
        else:
            print("  Status            : WARNING - No vectors created")

        return count > 0

    def get_document_count(self) -> int:
        """Return the number of vectors in the store."""
        if self.vectorstore is None:
            return 0
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
