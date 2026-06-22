"""
Document Service Module
───────────────────────
Provides a high-level API facade for the rest of the application
to interact with documents, hiding the complexities of GridFS and Metadata.
"""

from pathlib import Path
from bson.objectid import ObjectId

from .connection import MongoConnection
from .gridfs_manager import GridFSManager
from .metadata import MetadataManager
from .uploader import DocumentUploader


class DocumentService:
    """
    High-level service for document management.
    Application code should use this class rather than interacting
    with GridFS or MetadataManager directly.
    """
    
    def __init__(self):
        # Ensure connection is established
        MongoConnection()
        
        self.gridfs_mgr = GridFSManager()
        self.metadata_mgr = MetadataManager()
        self.uploader = DocumentUploader(self.gridfs_mgr, self.metadata_mgr)

    def upload_document(self, file_path: str | Path, title: str, category: str, **kwargs) -> ObjectId:
        """
        Upload a new medical document.
        """
        kwargs["title"] = title
        kwargs["category"] = category
        return self.uploader.upload(file_path, **kwargs)

    def list_documents(self) -> list[dict]:
        """
        Get metadata for all documents.
        """
        return self.metadata_mgr.list_all()

    def get_document(self, doc_id: str | ObjectId) -> dict | None:
        """
        Get metadata for a specific document.
        """
        return self.metadata_mgr.get_metadata(doc_id)

    def get_pending_documents(self) -> list[dict]:
        """
        Get all documents that need to be indexed.
        """
        return self.metadata_mgr.get_unindexed()

    def mark_document_indexed(self, doc_id: str | ObjectId, chunk_count: int) -> bool:
        """
        Mark a document as successfully indexed.
        """
        return self.metadata_mgr.mark_indexed(doc_id, chunk_count)

    def download_document_to_temp(self, doc_id: str | ObjectId) -> Path:
        """
        Download a document's PDF to a temporary file.
        Used by the indexer.
        """
        doc_metadata = self.get_document(doc_id)
        if not doc_metadata:
            raise ValueError(f"Document not found: {doc_id}")
            
        gridfs_id = doc_metadata.get("gridfs_id")
        if not gridfs_id:
            raise ValueError(f"Document {doc_id} is missing GridFS reference.")
            
        return self.gridfs_mgr.download_to_temp(gridfs_id, filename=doc_metadata.get("filename"))

    def delete_document(self, doc_id: str | ObjectId) -> bool:
        """
        Delete a document entirely (Metadata + GridFS).
        Note: This does NOT delete it from ChromaDB. That requires a ChromaDB update.
        """
        doc_metadata = self.get_document(doc_id)
        if not doc_metadata:
            return False
            
        # Delete from GridFS
        gridfs_id = doc_metadata.get("gridfs_id")
        if gridfs_id:
            self.gridfs_mgr.delete_file(gridfs_id)
            
        # Delete Metadata
        return self.metadata_mgr.delete_metadata(doc_id)

    def reset_indexing_status(self) -> int:
        """
        Reset all documents to unindexed status.
        Used during a full rebuild.
        """
        return self.metadata_mgr.reset_all_indexed_status()
