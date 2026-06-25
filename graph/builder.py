from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from .state import HealthAssistantState
from .nodes import WorkflowNodes
from .edges import add_graph_edges

def build_graph(llm, memory, rag):
    """
    Constructs and compiles the LangGraph workflow.
    """
    workflow = StateGraph(HealthAssistantState)
    nodes = WorkflowNodes(llm=llm, memory=memory, rag=rag)
    
    # Add Nodes
    workflow.add_node("input_node", nodes.input_node)
    workflow.add_node("memory_node", nodes.memory_node)
    workflow.add_node("intent_detection_node", nodes.intent_detection_node)
    workflow.add_node("retriever_node", nodes.retriever_node)
    workflow.add_node("prompt_builder_node", nodes.prompt_builder_node)
    workflow.add_node("llm_node", nodes.llm_node)
    workflow.add_node("general_response_node", nodes.general_response_node)
    workflow.add_node("citation_formatter_node", nodes.citation_formatter_node)
    
    # Set Entry Point
    workflow.add_edge(START, "input_node")
    
    # Add Edges and Routing
    add_graph_edges(workflow)
    
    # State Persistence Checkpointer
    checkpointer = MemorySaver()
    
    # Compile
    return workflow.compile(checkpointer=checkpointer)
