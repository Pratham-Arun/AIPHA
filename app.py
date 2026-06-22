import sys
import os
from pathlib import Path
import config
from database import MongoConnection, DocumentService
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from prompts import health_assistant_prompt
from memory import MemoryManager
from rag import RAGPipeline


def main():
    print("=" * 60)
    print("      AI-Powered Health Assistant      ")
    print("=" * 60)

    # ── Validate Configuration ──
    try:
        config.validate_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # ── Initialize MongoDB ──
    try:
        MongoConnection.validate_connection()
    except Exception as e:
        print(f"MongoDB Connection Error: {e}")
        sys.exit(1)

    # ── Initialize Components ──
    print("\nInitializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY
    )

    # ── Initialize Memory ──
    print("Initializing Memory Manager...")
    memory = MemoryManager()

    # ── Initialize Document Service ──
    doc_service = DocumentService()

    # ── Initialize RAG Pipeline ──
    rag = RAGPipeline()
    rag.initialize()

    # ── Create the Chain ──
    # Prompt (system + chat_history + context + question) → Gemini → Parser
    chain = health_assistant_prompt | llm | StrOutputParser()

    print("Chatbot is ready! Ask your health-related questions.")
    print("Commands:")
    print("  'upload'  - Upload a new medical PDF")
    print("  'docs'    - List all stored medical documents")
    print("  'rebuild' - Completely rebuild the document index")
    print("  'clear'   - Reset conversation memory")
    print("  'exit'    - Terminate the program\n")

    while True:
        try:
            # ── Step 1: Get user input ──
            user_input = input("User: ").strip()
            if not user_input:
                continue

            # ── Handle special commands ──
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
                
            if user_input.lower() == "docs":
                docs = doc_service.list_documents()
                if not docs:
                    print("Assistant: No documents found in the database.\n")
                else:
                    print(f"\nAssistant: Found {len(docs)} document(s):")
                    for doc in docs:
                        status = "Indexed" if doc.get("indexed") else "Pending"
                        print(f"  - {doc.get('title')} ({doc.get('filename')}) [{status}]")
                    print()
                continue
                
            if user_input.lower() == "upload":
                print("\n── Upload New Medical Document ──")
                file_path = input("File path (PDF): ").strip()
                
                if not file_path or not os.path.exists(file_path):
                    print("Error: Invalid file path.\n")
                    continue
                    
                title = input("Document Title: ").strip() or Path(file_path).stem
                category = input("Category (e.g., Diabetes, General): ").strip() or "General"
                
                try:
                    doc_service.upload_document(
                        file_path=file_path,
                        title=title,
                        category=category,
                        source="User Upload",
                        version="1.0"
                    )
                    # Automatically trigger incremental indexer
                    rag.index_new()
                except Exception as e:
                    print(f"Upload failed: {e}")
                print()
                continue

            # ── Step 2: Gather context from Memory and RAG ──
            # Get conversation history from memory
            chat_history = memory.get_formatted_history()

            # Retrieve relevant medical documents via RAG
            context = rag.retrieve(user_input)

            print("Assistant: Thinking...", end="\r")

            # ── Step 3: Invoke the Chain ──
            # The prompt template receives all three variables:
            #   - chat_history: previous conversation turns
            #   - context: relevant medical document chunks
            #   - question: the current user question
            response = chain.invoke({
                "question": user_input,
                "chat_history": chat_history,
                "context": context,
            })

            # ── Step 4: Save to Memory ──
            # Store this conversation turn for future context
            memory.save_conversation(user_input, response)

            # ── Step 5: Display the response ──
            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\nExiting Health Assistant. Stay healthy!")
            break
        except Exception as e:
            print(f"\nError occurred: {e}\n")


if __name__ == "__main__":
    main()
