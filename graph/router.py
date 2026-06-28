"""
Router – Phase 7 Multi-Agent Architecture
──────────────────────────────────────────
Routing functions that determine which nodes execute next
based on the current workflow state.

Flow:
  input_node
    → intent_detection_node
      → supervisor_node
        → route_after_supervisor
            ├─ needs_retrieval=True  → ["memory_node", "retriever_node"]  (parallel)
            └─ needs_retrieval=False → ["memory_node"]
          → agent_execution_node
            → formatter_node
              → END
"""

from typing import List, Literal
from .state import HealthAssistantState


def route_after_supervisor(state: HealthAssistantState) -> List[str]:
    """
    Routes after the Supervisor has made its decision.

    If the selected agent requires document retrieval, memory and retriever
    run in parallel. Otherwise only memory is loaded.
    """
    if state.get("needs_retrieval", False):
        return ["memory_node", "retriever_node"]
    return ["memory_node"]


def route_confidence(
    state: HealthAssistantState,
) -> Literal["agent_execution_node", "agent_execution_node"]:
    """
    Legacy confidence-based routing — now simplified.

    In Phase 7, every path converges on agent_execution_node regardless of
    confidence.  The Document QA Agent itself handles the low-confidence
    fallback by acknowledging when documents don't contain the answer.
    This function is kept for backward edge-registration compatibility.
    """
    return "agent_execution_node"


# ── Legacy alias (kept so old imports don't break) ───────────────────────────
def route_intent_parallel(state: HealthAssistantState) -> List[str]:
    """
    Legacy intent-based parallel routing.
    Delegates to route_after_supervisor for Phase 7 compatibility.
    """
    return route_after_supervisor(state)
