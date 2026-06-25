import operator
from typing import TypedDict, List, Any, Optional, Annotated

def merge_metrics(left: dict, right: dict) -> dict:
    """Merges two dictionaries containing metrics."""
    merged = left.copy()
    merged.update(right)
    return merged

class HealthAssistantState(TypedDict):
    """
    Represents the state of the LangGraph workflow.
    """
    query: str
    chat_history: str
    intent: str
    retrieved_docs: List[Any]
    context: str
    response: str
    sources: List[Any]
    confidence: str
    filter_dict: Optional[dict]
    
    # State flags and metadata
    error: str
    metrics: Annotated[dict, merge_metrics]
