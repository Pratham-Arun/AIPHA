"""
Migrate Local Database
──────────────────────
Uploads medical PDFs to Local MongoDB -> GridFS -> Metadata.
"""

import os
import sys
from pathlib import Path

# Ensure the root directory is in the sys path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database import DocumentService
import config

def migrate():
    print("====================================")
    print("Migration Script")
    print(f"Environment: {config.APP_ENV}")
    print(f"Target Database URI: {config.MONGODB_URI}")
    print("====================================")
    
    if "localhost" not in config.MONGODB_URI and "127.0.0.1" not in config.MONGODB_URI:
        print("Warning: This script is intended for Local MongoDB, but URI points to Atlas.")
        confirm = input("Continue anyway? (y/N): ")
        if confirm.lower() != 'y':
            print("Migration aborted.")
            return

    docs_dir = config.MEDICAL_DOCS_DIR
    if not docs_dir.exists() or not docs_dir.is_dir():
        print(f"Error: Directory '{docs_dir}' not found.")
        return

    doc_service = DocumentService()
    
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}")
        return
        
    print(f"Found {len(pdf_files)} PDF(s) to migrate.\n")
    
    for file_path in pdf_files:
        try:
            print(f"Migrating: {file_path.name}")
            doc_service.upload_document(
                file_path=str(file_path),
                title=file_path.stem,
                category="General",
                organization="General",
                source="Local Migration",
                version="1.0"
            )
        except Exception as e:
            print(f"Failed to migrate {file_path.name}: {e}")
            
    print("\nMigration Complete.")

if __name__ == "__main__":
    migrate()
