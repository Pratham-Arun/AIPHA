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
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY
    )

    # -- Initialize Memory --
    memory = MemoryManager()

    # -- Initialize Document Service --
    doc_service = DocumentService()

    # -- Initialize RAG Pipeline --
    rag = RAGPipeline()
    rag.initialize()

    print("GridFS\nReady")
    print("ChromaDB\nReady")
    print("Gemini\nReady\n")

    # -- Create the LangGraph Workflow --
    from graph.workflow import HealthAssistantWorkflow
    from graph.monitor import run_startup_diagnostics
    from graph.visualizer import export_graph_visualization
    
    print("Building LangGraph Workflow...")
    workflow = HealthAssistantWorkflow(llm=llm, memory=memory, rag=rag)
    run_startup_diagnostics(workflow)
    export_graph_visualization(workflow)

    print("Chatbot is ready! Ask your health-related questions.")
    print("Commands:")
    print("  'upload'       - Upload a new medical PDF")
    print("  'docs'         - List all stored medical documents")
    print("  'rebuild'      - Completely rebuild the document index")
    print("  'verify_index' - Verify system health and index status")
    print("  'clear'        - Reset conversation memory")
    print("  'exit'         - Terminate the program\n")

    try:
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
                    print("\n-- Index Verification --")
                    rag.verify_index()
                    print()
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
                        rag.index_new()
                    except Exception as e:
                        print(f"Upload failed: {e}")
                    print()
                    continue

                # -- Step 2: Invoke LangGraph Workflow --
                print("\nProcessing via LangGraph...")
                final_state = workflow.invoke(user_input)

                intent = final_state.get("intent", "Unknown")
                print(f"Detected Intent: {intent}")

                if intent == "Medical":
                    results = final_state.get("retrieved_docs", [])
                    print(f"Retrieved Context Chunks: {len(results)}")

                    if results:
                        print("\n-----------------------------------")
                        for doc, score in final_state.get("sources", []):
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")

                            title = doc.metadata.get("title", "")
                            if not title or title in ("Unknown", "Unknown Document"):
                                source_name = source.split("\\")[-1].split("/")[-1]
                                title = source_name.rsplit(".", 1)[0] if "." in source_name else source_name

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
                else:
                    print("Bypassing Retrieval (General/Greeting/Emergency Query)...")

                response = final_state.get("response", "")
                gen_time_s = final_state.get("processing_time", 0.0)

                memory.save_conversation(user_input, response)

                print(f"\n[Processing Time: {gen_time_s:.1f} sec]")
                print(f"Assistant: {response}\n")

            except KeyboardInterrupt:
                print("\nExiting Health Assistant. Stay healthy!")
                break
            except Exception as e:
                print(f"\nError occurred: {e}\n")
    finally:
        rag.shutdown()


if __name__ == "__main__":
    main()
