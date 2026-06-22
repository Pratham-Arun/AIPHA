"""
Uploader Module
───────────────
Handles the workflow of validating a PDF file, storing it in GridFS,
and generating its metadata record.
"""

import os
from pathlib import Path
from bson.objectid import ObjectId
from .gridfs_manager import GridFSManager
from .metadata import MetadataManager


class DocumentUploader:
    """
    Orchestrates the upload of a new medical document.
    Coordinates between GridFS and the Metadata collection.
    """
    
    def __init__(self, gridfs_mgr: GridFSManager, metadata_mgr: MetadataManager):
        self.gridfs_mgr = gridfs_mgr
        self.metadata_mgr = metadata_mgr

    def upload(self, file_path: str | Path, **metadata_kwargs) -> ObjectId:
        """
        Validate and upload a PDF document.
        
        Args:
            file_path: Path to the local PDF file.
            **metadata_kwargs: Additional metadata (title, category, source, etc.)
            
        Returns:
            The ObjectId of the new metadata document.
        """
        file_path = Path(file_path)
        
        # 1. Validation
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported. Got: {file_path.suffix}")
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("File is empty.")

        filename = file_path.name
        
        # Check if file with same name already exists in metadata
        # (Basic deduplication, can be expanded)
        existing_docs = self.metadata_mgr.list_all()
        if any(doc.get("filename") == filename for doc in existing_docs):
            print(f"Warning: A document with filename '{filename}' already exists.")
            # Decide whether to raise an error or allow duplicates
            # For now, we allow it but print a warning.

        print(f"  Uploading '{filename}' ({file_size / 1024 / 1024:.2f} MB)...")

        # 2. Store in GridFS
        # We store minimal metadata in GridFS itself, rely on the documents collection
        gridfs_metadata = {
            "original_filename": filename,
            "category": metadata_kwargs.get("category", "General")
        }
        gridfs_id = self.gridfs_mgr.upload_file(file_path, filename, gridfs_metadata)

        # 3. Store in Metadata Collection
        doc_info = {
            "filename": filename,
            "file_size_bytes": file_size,
            "gridfs_id": gridfs_id,
            **metadata_kwargs
        }
        
        try:
            doc_id = self.metadata_mgr.create_metadata(doc_info)
            print(f"  Upload successful! Document ID: {doc_id}")
            return doc_id
        except Exception as e:
            # Rollback GridFS upload if metadata fails
            print(f"  Failed to save metadata. Rolling back GridFS upload...")
            self.gridfs_mgr.delete_file(gridfs_id)
            raise e
