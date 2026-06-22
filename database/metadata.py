"""
Metadata Manager Module
───────────────────────
Handles CRUD operations for the 'documents' collection in MongoDB,
which stores metadata about uploaded PDFs.
"""

from datetime import datetime, timezone
from bson.objectid import ObjectId
from pymongo.collection import Collection
from .connection import MongoConnection


class MetadataManager:
    """
    Manages document metadata in the 'documents' MongoDB collection.
    """
    
    def __init__(self):
        db = MongoConnection().get_database()
        self.collection: Collection = db["documents"]
        # Create an index on the 'indexed' field for faster querying of pending documents
        self.collection.create_index("indexed")

    def create_metadata(self, doc_info: dict) -> ObjectId:
        """
        Insert a new document metadata record.
        
        Args:
            doc_info: Dictionary containing document metadata.
                      Should include: title, filename, category, source,
                      author, language, version, gridfs_id, file_size_bytes.
                      
        Returns:
            The ObjectId of the inserted metadata document.
        """
        document = {
            "title": doc_info.get("title", "Unknown"),
            "filename": doc_info.get("filename", "Unknown"),
            "category": doc_info.get("category", "General"),
            "source": doc_info.get("source", "Unknown"),
            "author": doc_info.get("author", "Unknown"),
            "language": doc_info.get("language", "English"),
            "version": doc_info.get("version", "1.0"),
            "upload_date": datetime.now(timezone.utc),
            "file_size_bytes": doc_info.get("file_size_bytes", 0),
            "indexed": False,
            "chunk_count": 0,
            "gridfs_id": doc_info.get("gridfs_id")
        }
        
        result = self.collection.insert_one(document)
        return result.inserted_id

    def get_metadata(self, doc_id: ObjectId | str) -> dict | None:
        """
        Retrieve metadata for a specific document.
        """
        if isinstance(doc_id, str):
            doc_id = ObjectId(doc_id)
        return self.collection.find_one({"_id": doc_id})

    def get_by_gridfs_id(self, gridfs_id: ObjectId | str) -> dict | None:
        """
        Find metadata document that points to a specific GridFS file.
        """
        if isinstance(gridfs_id, str):
            gridfs_id = ObjectId(gridfs_id)
        return self.collection.find_one({"gridfs_id": gridfs_id})

    def get_unindexed(self) -> list[dict]:
        """
        Find all documents that have not been indexed by RAG yet.
        """
        cursor = self.collection.find({"indexed": False})
        return list(cursor)

    def list_all(self) -> list[dict]:
        """
        List all document metadata, sorted by upload date descending.
        """
        cursor = self.collection.find().sort("upload_date", -1)
        return list(cursor)

    def mark_indexed(self, doc_id: ObjectId | str, chunk_count: int) -> bool:
        """
        Update a document to mark it as indexed and store its chunk count.
        """
        if isinstance(doc_id, str):
            doc_id = ObjectId(doc_id)
            
        result = self.collection.update_one(
            {"_id": doc_id},
            {"$set": {"indexed": True, "chunk_count": chunk_count}}
        )
        return result.modified_count > 0

    def reset_all_indexed_status(self) -> int:
        """
        Reset 'indexed' to False for all documents. Used during a full rebuild.
        Returns the number of documents updated.
        """
        result = self.collection.update_many(
            {},
            {"$set": {"indexed": False, "chunk_count": 0}}
        )
        return result.modified_count

    def delete_metadata(self, doc_id: ObjectId | str) -> bool:
        """
        Delete a document's metadata record.
        """
        if isinstance(doc_id, str):
            doc_id = ObjectId(doc_id)
            
        result = self.collection.delete_one({"_id": doc_id})
        return result.deleted_count > 0
