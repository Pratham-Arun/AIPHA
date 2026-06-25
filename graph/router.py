from typing import List, Literal
from .state import HealthAssistantState

def route_intent_parallel(state: HealthAssistantState) -> List[str]:
    """
    Routes the workflow based on intent.
    Runs memory and retriever in parallel for medical queries.
    """
    intent = state.get("intent", "General Chat")
    
    if intent in ["Medical Question", "Medical Report Analysis", "Nutrition", "Medicine Information"]:
        return ["memory_node", "retriever_node"]
    else:
        return ["memory_node"]

def route_confidence(state: HealthAssistantState) -> Literal["llm_node", "general_response_node"]:
    """
    Routes based on retriever confidence.
    """
    intent = state.get("intent", "General Chat")
    if intent not in ["Medical Question", "Medical Report Analysis", "Nutrition", "Medicine Information"]:
        # Bypass confidence check for non-medical queries
        return "llm_node"

    confidence = state.get("confidence", "Low")
    retrieved_docs = state.get("retrieved_docs", [])
    
    if len(retrieved_docs) == 0 and confidence == "Low":
        return "general_response_node"
    else:
        return "llm_node"
