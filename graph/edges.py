"""
Graph Edges – Phase 7.1  (Multi-Agent + Tool Layer)
─────────────────────────────────────────────────────
Full execution flow:

  START
    → input_node
      → intent_detection_node
        → supervisor_node
          → [conditional: route_after_supervisor]
              ├─ (needs_retrieval) → memory_node + retriever_node  (parallel)
              └─ (no retrieval)   → memory_node
          → tool_node              ← NEW Phase 7.1 (fan-in point)
            → agent_execution_node
              → formatter_node
                → END
"""

from langgraph.graph import StateGraph, END
from .router import route_after_supervisor


def add_graph_edges(workflow: StateGraph) -> None:
    """
    Registers all edges and conditional edges on the compiled StateGraph.
    """

    # 1. Input → Intent Detection
    workflow.add_edge("input_node", "intent_detection_node")

    # 2. Intent Detection → Supervisor
    workflow.add_edge("intent_detection_node", "supervisor_node")

    # 3. Supervisor → Memory (always) + Retriever (when retrieval needed), parallel
    workflow.add_conditional_edges(
        "supervisor_node",
        route_after_supervisor,
        {
            "memory_node":    "memory_node",
            "retriever_node": "retriever_node",
        },
    )

    # 4. Memory → Tool Node  (fan-in: waits for all parallel branches)
    workflow.add_edge("memory_node", "tool_node")

    # 5. Retriever → Tool Node  (parallel branch merges here)
    workflow.add_edge("retriever_node", "tool_node")

    # 6. Tool Node → Agent Execution
    workflow.add_edge("tool_node", "agent_execution_node")

    # 7. Agent Execution → Formatter
    workflow.add_edge("agent_execution_node", "formatter_node")

    # 8. Formatter → END
    workflow.add_edge("formatter_node", END)
