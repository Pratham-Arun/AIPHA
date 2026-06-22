"""
RAG Pipeline Module
───────────────────
Orchestrates the full RAG pipeline:
  1. Load or create vector store
  2. Index new documents from MongoDB incrementally
  3. Create retriever
  4. Retrieve relevant context for a query
"""

from .vectorstore import load_vectorstore, delete_vectorstore, get_or_create_vectorstore
from .retriever import get_retriever, format_retrieved_docs
from .indexer import Indexer
from database import DocumentService


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
        print("\n── Initializing RAG Pipeline ──")

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

    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant medical context for a query.
        """
        if not self.is_initialized or self.retriever is None:
            return "No medical documents loaded."

        # If vectorstore is empty, retriever will throw an error or return nothing
        if self.get_document_count() == 0:
            return "No medical documents loaded in the database."

        docs = self.retriever.invoke(query)
        return format_retrieved_docs(docs)

    def rebuild(self) -> bool:
        """
        Force rebuild the entire vector store.
        Deletes ChromaDB, resets all MongoDB documents to unindexed,
        and runs the indexer again.
        """
        print("\n── Rebuilding RAG Pipeline ──")
        
        # 1. Delete ChromaDB
        delete_vectorstore()
        
        # 2. Reset MongoDB indexed status
        doc_service = DocumentService()
        reset_count = doc_service.reset_indexing_status()
        print(f"  Reset indexing status for {reset_count} document(s) in MongoDB.")
        
        # 3. Re-initialize pipeline (creates new ChromaDB and runs indexer)
        self.is_initialized = False
        return self.initialize()

    def get_document_count(self) -> int:
        """Return the number of vectors in the store."""
        if self.vectorstore is None:
            return 0
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
