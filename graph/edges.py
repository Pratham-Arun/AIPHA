from langgraph.graph import StateGraph, END
from .router import route_intent_parallel, route_confidence

def add_graph_edges(workflow: StateGraph):
    """
    Defines the execution flow by adding standard and conditional edges to the graph.
    """
    # 1. Input flows to Intent Detection
    workflow.add_edge("input_node", "intent_detection_node")
    
    # 2. Intent Detection routes to Memory (and Retriever if Medical) in parallel
    workflow.add_conditional_edges(
        "intent_detection_node",
        route_intent_parallel,
        {
            "memory_node": "memory_node",
            "retriever_node": "retriever_node"
        }
    )
    
    # 3. Memory and Retriever flow to Prompt Builder
    workflow.add_edge("memory_node", "prompt_builder_node")
    workflow.add_edge("retriever_node", "prompt_builder_node")
    
    # 4. Prompt Builder routes based on Confidence
    workflow.add_conditional_edges(
        "prompt_builder_node",
        route_confidence,
        {
            "llm_node": "llm_node",
            "general_response_node": "general_response_node"
        }
    )
    
    # 5. LLM or General Response flow to Citation Formatter
    workflow.add_edge("llm_node", "citation_formatter_node")
    workflow.add_edge("general_response_node", "citation_formatter_node")
    
    # 6. End Workflow
    workflow.add_edge("citation_formatter_node", END)
