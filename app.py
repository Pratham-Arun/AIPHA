import sys
import os
from pathlib import Path
import time
import config
from database import MongoConnection, DocumentService
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from prompts import health_assistant_prompt
from memory import MemoryManager
from rag import RAGPipeline


def get_confidence_label(distance: float) -> str:
    """Map a ChromaDB distance score to a human-readable confidence label."""
    if distance < 0.8:
        return "High"
    elif distance < 1.1:
        return "Medium"
    else:
        return "Low"


def run_verify_index(doc_service: DocumentService, rag: RAGPipeline):
    """Run the verify_index health check command."""
    import gridfs
    from database.connection import MongoConnection as MC
    
    all_docs = doc_service.list_documents()
    total_docs = len(all_docs)
    
    # Calculate indexed vs pending
    indexed_docs = sum(1 for d in all_docs if d.get("indexed"))
    pending_docs = total_docs - indexed_docs
    
    # Calculate expected chunks based on what MongoDB thinks was successfully embedded
    expected_chunks = sum(d.get("chunk_count", 0) for d in all_docs)
    
    # Identify ghost documents (marked indexed but have 0 chunks)
    ghost_docs = sum(1 for d in all_docs if d.get("indexed") and d.get("chunk_count", 0) == 0)
    
    # Count GridFS files
    db = MC().get_database()
    fs = gridfs.GridFS(db)
    gridfs_count = db["fs.files"].count_documents({})
    
    vector_count = rag.get_document_count()
    
    print("\n-- Index Verification --")
    print(f"  MongoDB Documents : {total_docs}")
    print(f"  GridFS Files      : {gridfs_count}")
    print(f"  Indexed Status    : {indexed_docs} indexed, {pending_docs} pending")
    if ghost_docs > 0:
        print(f"  Ghost Documents   : {ghost_docs} (marked indexed but 0 chunks)")
    print(f"  Expected Chunks   : {expected_chunks}")
    print(f"  Vector Chunks     : {vector_count}")
    
    # Health check
    if pending_docs == 0 and vector_count == expected_chunks and ghost_docs == 0:
        print(f"  Status            : Healthy")
    elif ghost_docs > 0:
        print(f"  Status            : WARNING - {ghost_docs} documents marked indexed without chunks. Run 'rebuild'.")
    elif pending_docs > 0:
        print(f"  Status            : {pending_docs} document(s) not yet indexed")
    elif vector_count != expected_chunks:
        print(f"  Status            : WARNING - Mismatch between MongoDB expected chunks ({expected_chunks}) and ChromaDB ({vector_count})")
    else:
        print(f"  Status            : WARNING - No vectors in ChromaDB")
    print()


def main():
    print("=" * 60)
    print("      AI-Powered Health Assistant      ")
    print("=" * 60)

    # -- Validate Configuration --
    try:
        config.validate_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # -- Initialize MongoDB --
    try:
        MongoConnection.validate_connection()
    except Exception as e:
        print(f"MongoDB Connection Error: {e}")
        sys.exit(1)

    # -- Initialize Components --
    print("\nInitializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY
    )

    # -- Initialize Memory --
    print("Initializing Memory Manager...")
    memory = MemoryManager()

    # -- Initialize Document Service --
    doc_service = DocumentService()

    # -- Initialize RAG Pipeline --
    rag = RAGPipeline()
    rag.initialize()

    # -- System Diagnostics --
    docs_count = len(doc_service.list_documents())
    
    # Get actual model directly from class instance instead of config string
    from rag.embeddings import get_embedding_model
    try:
        actual_emb = get_embedding_model()
        emb_name = actual_emb.model_name
    except Exception:
        emb_name = config.EMBEDDING_MODEL

    print("\n-- System Diagnostics --")
    print(f"  MongoDB          : {docs_count} Documents")
    print(f"  GridFS           : {docs_count} Files")
    print(f"  Embedding Model  : {emb_name}")
    print(f"  Vector Store     : {rag.get_document_count()} Chunks")
    print(f"  Retriever        : Top K = {config.RETRIEVER_TOP_K}")
    print(f"  Distance Thresh  : {config.RETRIEVER_DISTANCE_THRESHOLD}")
    print(f"  LLM              : {config.LLM_MODEL}")
    print("  Memory           : Enabled")
    print("------------------------\n")

    # -- Create the Chain --
    chain = health_assistant_prompt | llm | StrOutputParser()

    print("Chatbot is ready! Ask your health-related questions.")
    print("Commands:")
    print("  'upload'       - Upload a new medical PDF")
    print("  'docs'         - List all stored medical documents")
    print("  'rebuild'      - Completely rebuild the document index")
    print("  'verify_index' - Verify system health and index status")
    print("  'clear'        - Reset conversation memory")
    print("  'exit'         - Terminate the program\n")

    while True:
        try:
            # -- Step 1: Get user input --
            user_input = input("User: ").strip()
            if not user_input:
                continue

            # -- Handle special commands --
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting Health Assistant. Stay healthy!")
                break

            if user_input.lower() == "clear":
                memory.clear_all()
                print("Assistant: Conversation memory cleared!\n")
                continue

            if user_input.lower() == "rebuild":
                rag.rebuild()
                continue
            
            if user_input.lower() == "verify_index":
                run_verify_index(doc_service, rag)
                continue
                
            if user_input.lower() == "docs":
                docs = doc_service.list_documents()
                if not docs:
                    print("Assistant: No documents found in the database.\n")
                else:
                    print(f"\nAssistant: Found {len(docs)} document(s):")
                    for doc in docs:
                        status = "Indexed" if doc.get("indexed") else "Pending"
                        org = doc.get("organization", "General")
                        print(f"  - {doc.get('title')} ({doc.get('filename')}) [{status}] [{org}]")
                    print()
                continue
                
            if user_input.lower() == "upload":
                print("\n-- Upload New Medical Document --")
                file_path = input("File path (PDF): ").strip()
                
                if not file_path or not os.path.exists(file_path):
                    print("Error: Invalid file path.\n")
                    continue
                    
                title = input("Document Title: ").strip() or Path(file_path).stem
                category = input("Category (e.g., Diabetes, General): ").strip() or "General"
                organization = input("Organization (e.g., WHO, CDC, NIH): ").strip() or "General"
                
                try:
                    doc_service.upload_document(
                        file_path=file_path,
                        title=title,
                        category=category,
                        organization=organization,
                        source="User Upload",
                        version="1.0"
                    )
                    # Automatically trigger incremental indexer
                    rag.index_new()
                except Exception as e:
                    print(f"Upload failed: {e}")
                print()
                continue

            # -- Step 2: Gather context from Memory and RAG --
            chat_history = memory.get_formatted_history()

            # Metadata Filtering Logic
            filter_dict = None
            input_lower = user_input.lower()
            if "who" in input_lower or "world health organization" in input_lower:
                filter_dict = {"source": {"$contains": "WHO"}}
            elif "cdc" in input_lower:
                filter_dict = {"source": {"$contains": "CDC"}}
            elif "nih" in input_lower:
                filter_dict = {"source": {"$contains": "NIH"}}

            print("\nSearching ChromaDB...")
            
            # Retrieve relevant medical documents via RAG
            context, results, discarded_count, search_time_ms = rag.retrieve(user_input, filter_dict=filter_dict)
            
            # -- Retrieval Validation --
            docs_retrieved_count = len(set(doc.metadata.get("source", "") for doc, _ in results))
            chunks_retrieved_count = len(results)
            
            print(f"Retrieved  : {chunks_retrieved_count + discarded_count} chunks")
            print(f"Relevant   : {chunks_retrieved_count}")
            print(f"Discarded  : {discarded_count}")
            print(f"Search Time: {search_time_ms} ms")
            
            if results:
                print("\n-----------------------------------")
                for doc, score in results:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    
                    # Determine clean title
                    title = doc.metadata.get("title", "")
                    if not title or title in ("Unknown", "Unknown Document"):
                        source_name = source.split("\\")[-1].split("/")[-1]
                        title = source_name.rsplit(".", 1)[0] if "." in source_name else source_name
                    
                    # Determine organization
                    org = doc.metadata.get("organization", "")
                    if not org or org == "General":
                        source_upper = source.upper()
                        for keyword in ["WHO", "CDC", "NIH", "FDA", "AHA"]:
                            if keyword in source_upper:
                                org = keyword
                                break
                        else:
                            org = "General"
                    
                    confidence = get_confidence_label(score)
                    
                    print(f"Organization: {org}")
                    print(f"Document    : {title}")
                    print(f"Page        : {page}")
                    print(f"Distance    : {score:.4f}")
                    print(f"Confidence  : {confidence}")
                    if confidence == "Low":
                        print("  * Low confidence retrieval. Results may not exactly match the question.")
                    print("-----------------------------------")
                print("\nRAG Context Found")
                print("Generating grounded answer...")
            else:
                print("\nNo Relevant Documents Found")
                print("Using Gemini General Knowledge...")

            gen_start_time = time.time()

            # -- Step 3: Invoke the Chain --
            response = chain.invoke({
                "question": user_input,
                "chat_history": chat_history,
                "context": context,
            })
            
            gen_time_s = time.time() - gen_start_time

            # -- Step 4: Save to Memory --
            memory.save_conversation(user_input, response)

            # -- Step 5: Display the response --
            print(f"\n[Generation Time: {gen_time_s:.1f} sec]")
            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\nExiting Health Assistant. Stay healthy!")
            break
        except Exception as e:
            print(f"\nError occurred: {e}\n")


if __name__ == "__main__":
    main()
