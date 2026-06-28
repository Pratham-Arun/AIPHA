import operator
from typing import TypedDict, List, Any, Optional, Annotated


def merge_metrics(left: dict, right: dict) -> dict:
    """Merges two metric dictionaries (right values overwrite left)."""
    merged = left.copy()
    merged.update(right)
    return merged


class HealthAssistantState(TypedDict):
    """
    Shared state for the multi-agent LangGraph workflow.

    Phase 7   : selected_agent, agent_type, needs_retrieval
    Phase 7.1 : tool_name, tool_result, calculator_output, doc_analysis_output
    Phase 7.2 : tool_result_obj (ToolResult), hybrid_search_metrics,
                request_log, avg_similarity
    """
    # ── Core input ───────────────────────────────────────────────────────────
    query: str
    chat_history: str

    # ── Routing / intent ─────────────────────────────────────────────────────
    intent: str
    selected_agent: str
    agent_type: str
    needs_retrieval: bool

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_docs: List[Any]
    context: str
    sources: List[Any]
    confidence: str
    avg_similarity: float          # Phase 7.2 – average similarity across chunks
    filter_dict: Optional[dict]

    # ── Tool Layer (Phase 7.1) ────────────────────────────────────────────────
    tool_name: str                 # kept for backward compat + formatter display
    tool_result: dict              # kept for backward compat
    calculator_output: str
    doc_analysis_output: str

    # ── Standardised ToolResult envelope (Phase 7.2) ─────────────────────────
    tool_result_obj: Any           # tools.tool_result.ToolResult instance

    # ── Hybrid Search metrics (Phase 7.2) ─────────────────────────────────────
    hybrid_search_metrics: dict    # vector_count, bm25_count, merged, dedup, final

    # ── Request-level structured log (Phase 7.2) ──────────────────────────────
    request_log: dict              # intent, agent, tool, timings, status

    # ── Response ──────────────────────────────────────────────────────────────
    response: str

    # ── State flags & metadata ────────────────────────────────────────────────
    error: str
    metrics: Annotated[dict, merge_metrics]
    processing_time: float
