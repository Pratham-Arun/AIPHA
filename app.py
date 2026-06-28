"""
AI-Powered Health Assistant  –  Phase 7.2
Multi-Agent Supervisor Architecture + Hybrid Search + Tool Layer
"""

import sys
import os
from pathlib import Path

import config
from database import MongoConnection, DocumentService
from langchain_google_genai import ChatGoogleGenerativeAI
from memory import MemoryManager
from rag import RAGPipeline

# ── Display labels ────────────────────────────────────────────────────────────
AGENT_DISPLAY = {
    "document_qa":    "📄 Document QA Agent",
    "drug_info":      "💊 Drug Information Agent",
    "nutrition":      "🥗 Nutrition Agent",
    "mental_health":  "🧠 Mental Health Agent",
    "general_health": "💬 General Health Agent",
}

TOOL_DISPLAY = {
    "knowledge_retrieval_tool":      "🔍 Knowledge Retrieval Tool",
    "retrieval_tool":                "🔍 Knowledge Retrieval Tool",
    "calculator_tool:bmi":           "🧮 Medical Calculator (BMI)",
    "calculator_tool:bmr":           "🧮 Medical Calculator (BMR)",
    "calculator_tool:water_intake":  "🧮 Medical Calculator (Water Intake)",
    "calculator_tool:ideal_weight":  "🧮 Medical Calculator (Ideal Weight)",
    "calculator_tool:bsa":           "🧮 Medical Calculator (BSA)",
    "document_analysis_tool":        "📋 Document Analysis Tool",
}


def _confidence_label(distance: float) -> str:
    if distance < 0.8:
        return "High"
    elif distance < 1.1:
        return "Medium"
    return "Low"


def main():
    print("=" * 62)
    print("  AI-Powered Health Assistant  –  Phase 7.2")
    print("  Multi-Agent + Hybrid Search + Tool Layer")
    print("=" * 62)

    # ── Validate config ───────────────────────────────────────────────────────
    try:
        config.validate_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # ── MongoDB ───────────────────────────────────────────────────────────────
    try:
        MongoConnection.validate_connection()
    except Exception as e:
        print(f"MongoDB Connection Error: {e}")
        sys.exit(1)

    # ── Core components ───────────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY,
    )
    memory      = MemoryManager()
    doc_service = DocumentService()
    rag         = RAGPipeline()
    rag.initialize()

    print("GridFS    : Ready")
    print("ChromaDB  : Ready")
    print("Gemini    : Ready\n")

    # ── Build workflow ────────────────────────────────────────────────────────
    from graph.workflow import HealthAssistantWorkflow
    from graph.monitor import run_startup_diagnostics
    from graph.visualizer import export_graph_visualization
    from tools.system_status_tool import SystemStatusTool

    print("Building Phase 7.2 Workflow (Hybrid Search + Tool Layer)...")
    workflow = HealthAssistantWorkflow(
        llm=llm, memory=memory, rag=rag, doc_service=doc_service
    )
    run_startup_diagnostics(workflow)
    export_graph_visualization(workflow)

    status_tool = SystemStatusTool(
        rag=rag, doc_service=doc_service, workflow=workflow, llm=llm
    )

    print("Chatbot is ready! Ask your health-related questions.")
    print("\nCommands:")
    print("  'upload'       - Upload a new medical PDF")
    print("  'docs'         - List all stored medical documents")
    print("  'status'       - Display full system diagnostics")
    print("  'rebuild'      - Completely rebuild the document index")
    print("  'verify_index' - Verify index health")
    print("  'clear'        - Reset conversation memory")
    print("  'exit'         - Terminate the program\n")

    try:
        while True:
            try:
                user_input = input("User: ").strip()
                if not user_input:
                    continue

                # ── Built-in commands ─────────────────────────────────────────

                if user_input.lower() in ("exit", "quit"):
                    print("Exiting Health Assistant. Stay healthy!")
                    break

                if user_input.lower() == "clear":
                    memory.clear_all()
                    print("Assistant: Conversation memory cleared!\n")
                    continue

                if user_input.lower() == "status":
                    print("\nRunning System Status Tool...")
                    print(status_tool.run())
                    print()
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
                            st  = "Indexed" if doc.get("indexed") else "Pending"
                            org = doc.get("organization", "General")
                            print(f"  - {doc.get('title')} ({doc.get('filename')}) [{st}] [{org}]")
                        print()
                    continue

                if user_input.lower() == "upload":
                    print("\n-- Upload New Medical Document --")
                    file_path = input("File path (PDF): ").strip()
                    if not file_path or not os.path.exists(file_path):
                        print("Error: Invalid file path.\n")
                        continue
                    title        = input("Document Title: ").strip() or Path(file_path).stem
                    category     = input("Category (e.g., Diabetes, General): ").strip() or "General"
                    organization = input("Organization (e.g., WHO, CDC, NIH): ").strip() or "General"
                    try:
                        doc_service.upload_document(
                            file_path=file_path, title=title,
                            category=category, organization=organization,
                            source="User Upload", version="1.0",
                        )
                        rag.index_new()
                        # Invalidate BM25 cache after new index
                        if hasattr(workflow, '_kr_tool'):
                            workflow._kr_tool.invalidate_bm25_cache()
                    except Exception as e:
                        print(f"Upload failed: {e}")
                    print()
                    continue

                # ── Invoke workflow ───────────────────────────────────────────
                print("\nProcessing via Multi-Agent + Hybrid Search LangGraph...")
                final_state = workflow.invoke(user_input)

                # ── Routing summary ───────────────────────────────────────────
                intent       = final_state.get("intent", "Unknown")
                agent_type   = final_state.get("agent_type", "general_health")
                needs_ret    = final_state.get("needs_retrieval", False)
                tool_name    = final_state.get("tool_name", "")
                confidence   = final_state.get("confidence", "")
                avg_sim      = final_state.get("avg_similarity", 0.0)
                agent_label  = AGENT_DISPLAY.get(agent_type, "🤖 AI Health Assistant")
                tool_label   = TOOL_DISPLAY.get(tool_name, "None")

                print(f"Detected Intent   : {intent}")
                print(f"Selected Agent    : {agent_label}")
                print(f"Tool Invoked      : {tool_label}")

                # ── Retrieval + hybrid search details ─────────────────────────
                if needs_ret:
                    results = final_state.get("retrieved_docs", [])
                    hsm     = final_state.get("hybrid_search_metrics", {})

                    # Hybrid search metrics summary line
                    if hsm:
                        bm25_used = hsm.get("bm25_used", False)
                        if bm25_used:
                            print(
                                f"Hybrid Search     : Vector={hsm.get('vector_count',0)} | "
                                f"BM25={hsm.get('bm25_count',0)} | "
                                f"Merged={hsm.get('merged_count',0)} | "
                                f"Dedup={hsm.get('duplicates_removed',0)} | "
                                f"Final={hsm.get('final_count',0)}"
                            )
                        else:
                            print(f"Hybrid Search     : Vector={hsm.get('vector_count',0)} (BM25 skipped)")

                    print(f"Retrieved Chunks  : {len(results)}")
                    if confidence:
                        print(f"Confidence        : {confidence}  |  Avg Similarity: {avg_sim:.4f}")

                    if results:
                        print("\n" + "-" * 44)
                        for doc, score in final_state.get("sources", []):
                            source = doc.metadata.get("source", "Unknown")
                            page   = doc.metadata.get("page", "?")
                            title  = doc.metadata.get("title", "")
                            if not title or title in ("Unknown", "Unknown Document"):
                                fname = source.replace("\\", "/").split("/")[-1]
                                title = fname.rsplit(".", 1)[0] if "." in fname else fname
                            org = doc.metadata.get("organization", "")
                            if not org or org == "General":
                                for kw in ["WHO", "CDC", "NIH", "FDA", "AHA"]:
                                    if kw in source.upper():
                                        org = kw
                                        break
                                else:
                                    org = "General"
                            conf = _confidence_label(score)
                            print(f"Organization : {org}")
                            print(f"Document     : {title}")
                            print(f"Page         : {page}")
                            print(f"Distance     : {score:.4f}  |  Confidence: {conf}")
                            print("-" * 44)
                        print("\nRAG Context Found — Generating grounded answer...")
                    else:
                        print("\nNo Relevant Documents Found — Using Gemini General Knowledge...")
                else:
                    print("Retrieval         : Skipped (not required for this query)")

                # ── Tool result previews ──────────────────────────────────────
                calc_out = final_state.get("calculator_output", "")
                doc_out  = final_state.get("doc_analysis_output", "")
                tr_obj   = final_state.get("tool_result_obj")

                if calc_out:
                    print(f"\n-- 🧮 Calculator Tool Result --\n{calc_out}")
                if doc_out:
                    print(f"\n-- 📋 Document Analysis Result --\n{doc_out}")

                # ToolResult log line (always shown for transparency)
                if tr_obj and not tr_obj.was_skipped:
                    print(f"\n-- Tool Execution Log --\n{tr_obj.log_line()}")

                # ── Response ──────────────────────────────────────────────────
                response = final_state.get("response", "")
                gen_time = final_state.get("processing_time", 0.0)
                memory.save_conversation(user_input, response)
                print(f"\n[Total Processing Time: {gen_time:.1f}s]")
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\nExiting Health Assistant. Stay healthy!")
                break
            except Exception as e:
                print(f"\nError occurred: {e}\n")
    finally:
        rag.shutdown()


if __name__ == "__main__":
    main()
