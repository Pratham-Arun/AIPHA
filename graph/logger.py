"""
Graph Logger  –  Phase 7.2
────────────────────────────
Structured logging for every node execution and tool invocation.

Phase 7.2 additions:
  - log_request_start / log_request_end  →  full request lifecycle
  - log_tool_execution                   →  per-tool structured line
  - log_hybrid_search                    →  hybrid search metrics line
"""

import logging
import time
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("HealthAssistant")


# ── Node-level logging ────────────────────────────────────────────────────────

def log_node_execution(
    node_name: str,
    execution_time: float,
    status: str,
    details: str = "",
) -> None:
    """Log execution of a LangGraph node."""
    msg = (
        f"Node: {node_name:<28} | "
        f"Status: {status:<10} | "
        f"Time: {execution_time*1000:.0f}ms"
    )
    if details:
        msg += f" | {details}"
    logger.info(msg)


def log_error(node_name: str, error: Exception) -> None:
    """Log an error from a LangGraph node."""
    logger.error(f"Node: {node_name} | ERROR: {error}", exc_info=True)


# ── Tool-level logging ────────────────────────────────────────────────────────

def log_tool_execution(tool_result) -> None:
    """
    Log a standardised ToolResult.

    Args:
        tool_result: tools.tool_result.ToolResult instance
    """
    logger.info(f"Tool Execution | {tool_result.log_line()}")


# ── Hybrid search metrics logging ────────────────────────────────────────────

def log_hybrid_search(result) -> None:
    """
    Log hybrid search metrics from a KnowledgeRetrievalResult.

    Args:
        result: tools.knowledge_retrieval_tool.KnowledgeRetrievalResult
    """
    logger.info(f"Hybrid Search  | {result.format_summary()}")


# ── Request lifecycle logging ─────────────────────────────────────────────────

def log_request_start(query: str) -> float:
    """
    Log the start of a new user request.  Returns the start timestamp.
    """
    logger.info(f"Request Started | Query: {query[:80]!r}")
    return time.time()


def log_request_end(
    start_time: float,
    intent: str,
    agent: str,
    tool: str,
    retrieval_ms: int = 0,
    llm_ms: int = 0,
) -> None:
    """
    Log the end of a completed request with full timing breakdown.

    Args:
        start_time:   time.time() value from log_request_start.
        intent:       Detected intent label.
        agent:        Selected agent type.
        tool:         Tool name used (empty string if none).
        retrieval_ms: Time spent in retrieval (ms).
        llm_ms:       Time spent in LLM call (ms).
    """
    total_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"Request Complete | "
        f"Intent: {intent} | "
        f"Agent: {agent} | "
        f"Tool: {tool or 'none'} | "
        f"Retrieval: {retrieval_ms}ms | "
        f"LLM: {llm_ms}ms | "
        f"Total: {total_ms}ms"
    )
