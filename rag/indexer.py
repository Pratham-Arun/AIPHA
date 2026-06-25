"""
Indexer Module
──────────────
Handles incremental indexing of new medical documents from MongoDB into ChromaDB.
"""

import hashlib
import os

from database import DocumentService
from .loader import load_single_pdf
from .splitter import split_documents
from .utils import assign_chunk_ids
from .vectorstore import VectorStore


class Indexer:
    """Performs incremental indexing of documents from MongoDB into ChromaDB."""

    def __init__(self, vector_store: VectorStore | None = None):
        self.doc_service = DocumentService()
        self.vector_store = vector_store or VectorStore()
        self.vector_store.initialize()

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
                print("    Downloading from GridFS...")
                temp_path = self.doc_service.download_document_to_temp(doc_id)

                with open(temp_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                existing_doc = self.doc_service.metadata_mgr.get_by_hash(file_hash)
                if existing_doc and existing_doc.get("indexed") and str(existing_doc.get("_id")) != str(doc_id):
                    print(f"    Skipped Duplicate: '{filename}'")
                    self.doc_service.mark_document_indexed(doc_id, 0, file_hash=file_hash)
                    indexed_count += 1
                    continue

                print("    Loading PDF...")
                documents = load_single_pdf(temp_path)

                for d in documents:
                    d.metadata["source"] = filename
                    d.metadata["doc_id"] = str(doc_id)

                print("    Splitting into chunks...")
                chunks = split_documents(documents)
                chunk_count = len(chunks)

                print("    Generating embeddings and storing in ChromaDB...")
                ids = assign_chunk_ids(chunks, doc_id)

                skipped = sum(1 for chunk_id in ids if self.vector_store.exists(chunk_id))
                if skipped:
                    print(f"    Skipping {skipped} existing chunk(s)...")

                added = self.vector_store.add_documents(chunks, ids)
                if added == 0 and skipped < chunk_count:
                    raise RuntimeError("Failed to insert any chunks into ChromaDB")

                print("    Updating metadata...")
                self.doc_service.mark_document_indexed(doc_id, chunk_count, file_hash=file_hash)

                print(f"  ✓ Successfully indexed '{filename}' ({chunk_count} chunks)")
                indexed_count += 1

            except Exception as e:
                print(f"  ✗ Failed to index '{filename}': {e}")
            finally:
                if temp_path and temp_path.exists():
                    try:
                        os.remove(temp_path)
                    except Exception as cleanup_err:
                        print(f"    Failed to clean up temp file {temp_path}: {cleanup_err}")

        print(f"\n── Indexing Complete: {indexed_count}/{len(pending_docs)} documents indexed ──")
        return indexed_count
