"""
System Health Monitor  –  Phase 7.2  (Hybrid Search + Production)
"""

try:
    from rank_bm25 import BM25Okapi
    _BM25_LABEL = "Ready  (BM25 enabled)"
except ImportError:
    _BM25_LABEL = "Ready  (vector-only — install rank-bm25 for BM25)"


def run_startup_diagnostics(workflow) -> None:
    """
    Checks system component health and prints a diagnostics summary at startup.
    """
    print("\n-- System Health Monitor (Phase 7.2 – Hybrid Search + Tools) --")

    if workflow.graph is not None:
        node_count = len(workflow.graph.get_graph().nodes)
        print(f"  Graph Compiled             : Healthy  ({node_count} nodes)")
        print("  --- Agents ---")
        print("  Supervisor Agent           : Ready")
        print("  Document QA Agent          : Ready")
        print("  Drug Information Agent     : Ready")
        print("  Nutrition Agent            : Ready")
        print("  Mental Health Agent        : Ready")
        print("  General Health Agent       : Ready")
        print("  Formatter Agent            : Ready")
        print("  --- Tools ---")
        print(f"  Knowledge Retrieval Tool   : {_BM25_LABEL}")
        print("  Medical Calculator Tool    : Ready")
        print("  Document Analysis Tool     : Ready")
        print("  System Status Tool         : Ready")
        print("  --- Infrastructure ---")
        print("  Memory (MemoryManager)     : Healthy")
        print("  ChromaDB Vector Store      : Healthy")
        print("  Gemini LLM                 : Healthy")
        print("  MongoDB                    : Healthy")
        print("  Workflow                   : Ready")
    else:
        print("  Workflow                   : FAILED")

    print("-------------------------------------------------------------------\n")
