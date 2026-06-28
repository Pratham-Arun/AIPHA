"""
Graph Builder  –  Phase 7.2  (Hybrid Search + Production Improvements)
────────────────────────────────────────────────────────────────────────
Constructs and compiles the Supervisor-driven multi-agent workflow
with the Phase 7.2 Tool Layer (Hybrid Search + ToolResult + structured logging).

Node registry:
  input_node            – parse filters, initialise metrics & request_log
  intent_detection_node – classify user query intent
  supervisor_node       – select specialised agent
  memory_node           – load conversation history
  retriever_node        – Hybrid Search via KnowledgeRetrievalTool
  tool_node             – deterministic tool execution (ToolResult wrapped)
  agent_execution_node  – dispatch to selected agent
  formatter_node        – format response + citations + hybrid metrics
"""

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from .state import HealthAssistantState
from .nodes import WorkflowNodes
from .edges import add_graph_edges


def build_graph(llm, memory, rag, doc_service=None):
    """
    Constructs and compiles the Phase 7.2 multi-agent LangGraph workflow.

    Args:
        llm:         Initialised Gemini LLM instance.
        memory:      Initialised MemoryManager instance.
        rag:         Initialised RAGPipeline instance.
        doc_service: Initialised DocumentService (used by DocumentAnalysisTool).

    Returns:
        Compiled LangGraph CompiledGraph with MemorySaver checkpointer.
    """
    workflow = StateGraph(HealthAssistantState)
    nodes    = WorkflowNodes(llm=llm, memory=memory, rag=rag, doc_service=doc_service)

    # ── Register Nodes ────────────────────────────────────────────────────────
    workflow.add_node("input_node",            nodes.input_node)
    workflow.add_node("intent_detection_node", nodes.intent_detection_node)
    workflow.add_node("supervisor_node",       nodes.supervisor_node)
    workflow.add_node("memory_node",           nodes.memory_node)
    workflow.add_node("retriever_node",        nodes.retriever_node)   # Hybrid Search
    workflow.add_node("tool_node",             nodes.tool_node)        # ToolResult wrapper
    workflow.add_node("agent_execution_node",  nodes.agent_execution_node)
    workflow.add_node("formatter_node",        nodes.formatter_node)   # hybrid metrics

    # ── Entry Point ───────────────────────────────────────────────────────────
    workflow.add_edge(START, "input_node")

    # ── Edges & Routing ───────────────────────────────────────────────────────
    add_graph_edges(workflow)

    # ── State Persistence ─────────────────────────────────────────────────────
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
