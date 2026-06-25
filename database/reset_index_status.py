"""
Utility script to reset the indexed flag and chunk count for all documents.
"""

import sys
from pathlib import Path

# Ensure the root directory is in the sys path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.document_service import DocumentService

def reset_index_status():
    print("====================================")
    print("Reset Metadata Status Script")
    print("====================================")
    doc_service = DocumentService()
    
    docs = doc_service.list_documents()
    total_docs = len(docs)
    print(f"Documents Found: {total_docs}")
    
    reset_count = doc_service.reset_indexing_status()
    
    print(f"Reset {reset_count} documents.")
    print(f"Indexed: 0")
    print(f"Pending: {total_docs}")

if __name__ == "__main__":
    reset_index_status()
