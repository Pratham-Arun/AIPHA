"""
Migration Utility
─────────────────
One-time script to migrate existing PDFs from the local 'Medical Documents'
folder into MongoDB GridFS and mark them as indexed.
"""

from pathlib import Path
import config
from .document_service import DocumentService

def migrate_existing_documents():
    """
    Migrate existing PDFs from the local directory into MongoDB.
    Since these are presumably already in ChromaDB, we mark them as indexed.
    """
    print("=" * 60)
    print("      MongoDB Document Migration Utility")
    print("=" * 60)

    docs_dir = config.MEDICAL_DOCS_DIR
    if not docs_dir.exists():
        print(f"Directory {docs_dir} does not exist. Nothing to migrate.")
        return

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}. Nothing to migrate.")
        return

    print(f"Found {len(pdf_files)} PDF files to migrate.")

    service = DocumentService()
    existing_docs = service.list_documents()
    existing_filenames = [doc.get("filename") for doc in existing_docs]

    migrated_count = 0
    skipped_count = 0

    for pdf in pdf_files:
        filename = pdf.name
        
        if filename in existing_filenames:
            print(f"  Skipping '{filename}' (already exists in MongoDB)")
            skipped_count += 1
            continue

        print(f"\nMigrating '{filename}'...")
        
        # Derive a basic title and category from the filename
        base_name = pdf.stem
        title = base_name.replace("-", " ").replace("_", " ").title()
        
        try:
            doc_id = service.upload_document(
                file_path=pdf,
                title=title,
                category="General",
                source="Migration",
                version="1.0"
            )
            
            # Document is created with indexed=false by default.
            # The incremental indexer will download from GridFS, chunk,
            # generate embeddings, insert into ChromaDB, and THEN mark indexed=true.
            
            print(f"  Successfully migrated! ID: {doc_id}")
            migrated_count += 1
            
        except Exception as e:
            print(f"  Error migrating '{filename}': {e}")

    print("\n" + "=" * 60)
    print(f"Migration Complete: {migrated_count} migrated, {skipped_count} skipped.")
    print("=" * 60)

if __name__ == "__main__":
    migrate_existing_documents()
