"""
Sync Atlas Database
───────────────────
Synchronizes Local MongoDB to Atlas.
"""

import os
import sys
from pathlib import Path
from pymongo import MongoClient
import gridfs

# Ensure the root directory is in the sys path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config

def sync_atlas():
    print("====================================")
    print("Sync Script: Local MongoDB -> Atlas")
    print("====================================")
    
    local_uri = "mongodb://localhost:27017"
    local_db_name = config.DATABASE_NAME
    
    # Load atlas uri from credentials or env
    atlas_uri = os.getenv("ATLAS_URI")
    if not atlas_uri:
        print("ATLAS_URI not found in environment.")
        atlas_uri = input("Please enter your MongoDB Atlas URI: ").strip()
        
    if not atlas_uri:
        print("Atlas URI is required. Aborting.")
        return
        
    print(f"\nSource: {local_uri}")
    print(f"Target: Atlas ({atlas_uri.split('@')[-1] if '@' in atlas_uri else 'hidden'})")
    
    try:
        local_client = MongoClient(local_uri, serverSelectionTimeoutMS=5000)
        local_client.admin.command('ping')
        local_db = local_client[local_db_name]
        
        atlas_client = MongoClient(atlas_uri, serverSelectionTimeoutMS=5000)
        atlas_client.admin.command('ping')
        atlas_db = atlas_client[local_db_name]
    except Exception as e:
        print(f"Connection failed: {e}")
        return
        
    # Sync metadata collection
    print("\nSyncing 'documents' metadata collection...")
    local_docs = list(local_db["documents"].find())
    print(f"Found {len(local_docs)} documents in local database.")
    
    if local_docs:
        # Clear remote first for a full sync
        atlas_db["documents"].delete_many({})
        atlas_db["documents"].insert_many(local_docs)
        print("Metadata synced.")
        
    # Sync GridFS
    print("\nSyncing GridFS files...")
    local_fs = gridfs.GridFS(local_db)
    atlas_fs = gridfs.GridFS(atlas_db)
    
    # Clear remote GridFS
    atlas_db["fs.files"].delete_many({})
    atlas_db["fs.chunks"].delete_many({})
    
    local_files = list(local_db["fs.files"].find())
    print(f"Found {len(local_files)} files in local GridFS.")
    
    for file_doc in local_files:
        try:
            file_id = file_doc["_id"]
            filename = file_doc.get("filename", "unknown")
            print(f"Syncing file: {filename}")
            
            # Read from local
            with local_fs.get(file_id) as f:
                data = f.read()
                
            # Write to atlas with same id
            atlas_fs.put(data, _id=file_id, filename=filename, metadata=file_doc.get("metadata", {}))
        except Exception as e:
            print(f"Failed to sync file {file_doc.get('filename')}: {e}")
            
    print("\nSync Complete.")

if __name__ == "__main__":
    sync_atlas()
