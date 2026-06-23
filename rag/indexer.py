"""
Indexer Module
──────────────
Handles incremental indexing of new medical documents from MongoDB into ChromaDB.
"""

from database import DocumentService
from .loader import load_single_pdf
from .splitter import split_documents
from .vectorstore import get_or_create_vectorstore
import os
import hashlib


class Indexer:
    """
    Service to perform incremental indexing of documents.
    """
    def __init__(self):
        self.doc_service = DocumentService()
        self.vectorstore = get_or_create_vectorstore()

    def index_new_documents(self) -> int:
        """
        Check MongoDB for unindexed documents, download them from GridFS,
        chunk them, embed them into ChromaDB, and mark them as indexed.
        
        Returns:
            Number of new documents successfully indexed.
        """
        print("\n── Checking for new documents to index ──")
        print("  Scanning MongoDB...")
        
        all_docs_count = len(self.doc_service.list_documents())
        pending_docs = self.doc_service.get_pending_documents()
        
        print(f"  Documents Found : {all_docs_count}")
        print(f"  Already Indexed : {all_docs_count - len(pending_docs)}")
        print(f"  Pending Index   : {len(pending_docs)}\n")
        
        if not pending_docs:
            print("  No new documents to index.")
            return 0
            
        indexed_count = 0
        
        for doc in pending_docs:
            doc_id = doc["_id"]
            filename = doc["filename"]
            
            print(f"\n  Processing '{filename}'...")
            
            temp_path = None
            try:
                # 1. Download PDF to temp file
                print("    Downloading from GridFS...")
                temp_path = self.doc_service.download_document_to_temp(doc_id)
                
                # Compute SHA256 Hash for duplicate detection
                with open(temp_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                existing_doc = self.doc_service.metadata_mgr.get_by_hash(file_hash)
                if existing_doc and existing_doc.get("indexed") and str(existing_doc.get("_id")) != str(doc_id):
                    print(f"    Skipped Duplicate: '{filename}'")
                    # Mark as indexed without re-embedding
                    self.doc_service.mark_document_indexed(doc_id, 0, file_hash=file_hash)
                    indexed_count += 1
                    continue
                
                # 2. Load PDF
                print("    Loading PDF...")
                documents = load_single_pdf(temp_path)
                
                # Update source metadata to the original filename instead of temp path
                for d in documents:
                    d.metadata["source"] = filename
                    # Add doc_id so we can trace chunks back to MongoDB
                    d.metadata["doc_id"] = str(doc_id)
                    
                # 3. Split into chunks
                print("    Splitting into chunks...")
                chunks = split_documents(documents)
                chunk_count = len(chunks)
                
                # 4. Add to ChromaDB
                print("    Generating embeddings and storing in ChromaDB...")
                self.vectorstore.add_documents(chunks)
                
                # 5. Mark as indexed
                print("    Updating metadata...")
                self.doc_service.mark_document_indexed(doc_id, chunk_count, file_hash=file_hash)
                
                print(f"  ✓ Successfully indexed '{filename}' ({chunk_count} chunks)")
                indexed_count += 1
                
            except Exception as e:
                print(f"  ✗ Failed to index '{filename}': {e}")
            finally:
                # Clean up temporary file
                if temp_path and temp_path.exists():
                    try:
                        os.remove(temp_path)
                    except Exception as cleanup_err:
                        print(f"    Failed to clean up temp file {temp_path}: {cleanup_err}")
                        
        print(f"\n── Indexing Complete: {indexed_count}/{len(pending_docs)} documents indexed ──")
        return indexed_count
