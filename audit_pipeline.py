"""
Pipeline Audit Script
---------------------
Probes the actual state of MongoDB documents and ChromaDB to identify
why 51 documents are marked indexed=true but only 5573 vectors exist.
"""
import sys
sys.path.append('.')
import os
os.environ["PYTHONUTF8"] = "1"

import config
config.validate_config()

from database import DocumentService, MongoConnection
from database.connection import MongoConnection as MC
import gridfs

print("=" * 60)
print("  PIPELINE AUDIT")
print("=" * 60)

# 1. Check MongoDB documents collection
print("\n--- 1. MongoDB 'documents' Collection ---")
ds = DocumentService()
all_docs = ds.list_documents()
print(f"Total documents: {len(all_docs)}")

indexed_true = [d for d in all_docs if d.get("indexed") == True]
indexed_false = [d for d in all_docs if d.get("indexed") == False]
indexed_missing = [d for d in all_docs if "indexed" not in d]

print(f"  indexed=true  : {len(indexed_true)}")
print(f"  indexed=false : {len(indexed_false)}")
print(f"  indexed field missing: {len(indexed_missing)}")

# 2. Check chunk_count values
print("\n--- 2. chunk_count Analysis ---")
total_expected_chunks = 0
zero_chunk_docs = []
for d in all_docs:
    cc = d.get("chunk_count", 0)
    total_expected_chunks += cc
    if cc == 0:
        zero_chunk_docs.append(d.get("filename"))

print(f"Sum of all chunk_count: {total_expected_chunks}")
print(f"Documents with chunk_count=0: {len(zero_chunk_docs)}")
if zero_chunk_docs:
    print(f"  First 10: {zero_chunk_docs[:10]}")

# 3. Check which docs have chunk_count > 0
print("\n--- 3. Documents with chunk_count > 0 ---")
has_chunks = [(d.get("filename"), d.get("chunk_count")) for d in all_docs if d.get("chunk_count", 0) > 0]
for fn, cc in has_chunks:
    print(f"  {fn}: {cc} chunks")
print(f"Total: {len(has_chunks)} documents with actual chunks")

# 4. Check GridFS
print("\n--- 4. GridFS ---")
db = MC().get_database()
fs = gridfs.GridFS(db)
gridfs_count = db["fs.files"].count_documents({})
print(f"GridFS files: {gridfs_count}")

# 5. Check ChromaDB
print("\n--- 5. ChromaDB ---")
from rag.vectorstore import VectorStore
vs = VectorStore()
vs.initialize()
chroma_count = vs.count()
print(f"ChromaDB vectors: {chroma_count}")

# 6. Sample a few ChromaDB docs to see what sources are in there
print("\n--- 6. ChromaDB Source Distribution ---")
try:
    result = vs._chroma._collection.get(include=["metadatas"], limit=100)
    sources = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"Unique sources in ChromaDB: {len(sources)}")
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:20]:
        print(f"  {src}: {count} vectors")
except Exception as e:
    print(f"Error sampling ChromaDB: {e}")

# 7. Verify the embedding model
print("\n--- 7. Embedding Model ---")
from rag.embeddings import get_embedding_model
emb = get_embedding_model()
print(f"Actual embedding class : {type(emb).__name__}")
print(f"Actual model_name      : {emb.model_name}")
print(f"config.EMBEDDING_MODEL : {config.EMBEDDING_MODEL}")
if "MiniLM" in emb.model_name or "minilm" in emb.model_name.lower():
    print("  >> Code uses all-MiniLM-L6-v2 (HuggingFace)")
else:
    print(f"  >> Code uses: {emb.model_name}")

if config.EMBEDDING_MODEL != emb.model_name:
    print(f"  ** MISMATCH: config says '{config.EMBEDDING_MODEL}' but code uses '{emb.model_name}'")

# 8. Check the source field of a migrated doc
print("\n--- 8. Migration Source Field ---")
for d in all_docs[:5]:
    print(f"  {d.get('filename')}: source='{d.get('source')}', indexed={d.get('indexed')}, chunks={d.get('chunk_count')}")

# 9. Summary
print("\n" + "=" * 60)
print("  DIAGNOSIS")
print("=" * 60)
print(f"MongoDB docs:             {len(all_docs)}")
print(f"Marked indexed=true:      {len(indexed_true)}")
print(f"With chunk_count > 0:     {len(has_chunks)}")
print(f"Expected total chunks:    {total_expected_chunks}")
print(f"Actual ChromaDB vectors:  {chroma_count}")

if len(indexed_true) > len(has_chunks):
    print(f"\n** ROOT CAUSE: {len(indexed_true) - len(has_chunks)} documents are marked indexed=true")
    print(f"   but have chunk_count=0, meaning they were marked indexed during migration")
    print(f"   WITHOUT actually generating embeddings.")
    print(f"   migrate.py line 67: service.mark_document_indexed(doc_id, chunk_count=0)")
    print(f"   This marks the doc as indexed=true with 0 chunks, so the indexer")
    print(f"   never picks them up again.")
