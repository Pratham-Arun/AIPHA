"""
GridFS Manager Module
─────────────────────
Handles binary storage of PDF files in MongoDB using GridFS.
"""

from pathlib import Path
import gridfs
from bson.objectid import ObjectId
from .connection import MongoConnection
import config


class GridFSManager:
    """
    Manages PDF file storage in MongoDB GridFS.
    """
    
    def __init__(self):
        db = MongoConnection().get_database()
        self.fs = gridfs.GridFS(db)

    def upload_file(self, file_path: str | Path, filename: str, metadata: dict = None) -> ObjectId:
        """
        Upload a file to GridFS.
        
        Args:
            file_path: Path to the local file.
            filename: Name to store the file under in GridFS.
            metadata: Optional dictionary of metadata to store alongside the file.
            
        Returns:
            The ObjectId of the stored file in GridFS.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = metadata or {}
        
        with open(file_path, "rb") as f:
            file_id = self.fs.put(
                f, 
                filename=filename, 
                metadata=metadata,
                content_type="application/pdf"
            )
            
        return file_id

    def download_file(self, gridfs_id: ObjectId | str) -> bytes:
        """
        Retrieve file content from GridFS as bytes.
        """
        if isinstance(gridfs_id, str):
            gridfs_id = ObjectId(gridfs_id)
            
        try:
            grid_out = self.fs.get(gridfs_id)
            return grid_out.read()
        except gridfs.errors.NoFile:
            raise FileNotFoundError(f"File not found in GridFS with id: {gridfs_id}")

    def download_to_temp(self, gridfs_id: ObjectId | str, filename: str = None) -> Path:
        """
        Download a file from GridFS to a temporary local file.
        Useful for components like PyPDFLoader that expect a file path.
        
        Returns:
            Path to the temporary file.
        """
        if isinstance(gridfs_id, str):
            gridfs_id = ObjectId(gridfs_id)
            
        try:
            grid_out = self.fs.get(gridfs_id)
            
            # Use original filename if not provided
            if not filename:
                filename = grid_out.filename or f"temp_{gridfs_id}.pdf"
                
            # Create a temp directory inside data/
            temp_dir = config.DATA_DIR / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            temp_path = temp_dir / filename
            
            with open(temp_path, "wb") as f:
                f.write(grid_out.read())
                
            return temp_path
            
        except gridfs.errors.NoFile:
            raise FileNotFoundError(f"File not found in GridFS with id: {gridfs_id}")

    def delete_file(self, gridfs_id: ObjectId | str) -> bool:
        """
        Delete a file from GridFS.
        """
        if isinstance(gridfs_id, str):
            gridfs_id = ObjectId(gridfs_id)
            
        try:
            self.fs.delete(gridfs_id)
            return True
        except gridfs.errors.NoFile:
            return False

    def file_exists(self, gridfs_id: ObjectId | str) -> bool:
        """
        Check if a file exists in GridFS.
        """
        if isinstance(gridfs_id, str):
            gridfs_id = ObjectId(gridfs_id)
        return self.fs.exists(gridfs_id)
