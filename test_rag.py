import sys
sys.path.append('.')
import os
os.environ["PYTHONUTF8"] = "1"

import config
config.validate_config()

from rag.pipeline import RAGPipeline

rag = RAGPipeline()
rag.initialize()

# Test 1: Relevant query - should keep chunks
print("\n=== Test 1: Relevant Query ===")
query = "What causes iron deficiency anemia?"
context, results, discarded, time_ms = rag.retrieve(query)
print(f"Relevant : {len(results)}")
print(f"Discarded: {discarded}")
print(f"Time     : {time_ms} ms")
if results:
    for doc, score in results:
        confidence = "High" if score < 0.8 else ("Medium" if score < 1.1 else "Low")
        print(f"  Distance: {score:.4f}  Confidence: {confidence}")

# Test 2: Unrelated query - should discard all
print("\n=== Test 2: Unrelated Query (Huntington's) ===")
query = "What is Huntington's disease?"
context, results, discarded, time_ms = rag.retrieve(query)
print(f"Relevant : {len(results)}")
print(f"Discarded: {discarded}")
print(f"Time     : {time_ms} ms")
if results:
    for doc, score in results:
        confidence = "High" if score < 0.8 else ("Medium" if score < 1.1 else "Low")
        print(f"  Distance: {score:.4f}  Confidence: {confidence}")
else:
    print("  -> All chunks discarded. General Knowledge mode would activate.")

# Test 3: verify_index simulation
print("\n=== Test 3: Index Status ===")
from database import DocumentService
ds = DocumentService()
all_docs = ds.list_documents()
total = len(all_docs)
indexed = sum(1 for d in all_docs if d.get("indexed"))
print(f"MongoDB Documents : {total}")
print(f"Indexed           : {indexed}")
print(f"Vector Chunks     : {rag.get_document_count()}")
