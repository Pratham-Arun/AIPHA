"""
Workflow Wrapper – Phase 7.1  (Multi-Agent + Tool Layer)
──────────────────────────────────────────────────────────
Thin wrapper around the compiled LangGraph graph.
Used by app.py to invoke the Supervisor-driven multi-agent workflow.
"""

from .builder import build_graph


class HealthAssistantWorkflow:
    """
    Wraps the compiled LangGraph workflow for convenient invocation from app.py.
    """

    def __init__(self, llm, memory, rag, doc_service=None):
        self.graph = build_graph(
            llm=llm, memory=memory, rag=rag, doc_service=doc_service
        )

    def invoke(self, query: str, thread_id: str = "default") -> dict:
        """
        Execute the multi-agent + tool-layer workflow for the given user query.

        Args:
            query:     The user's input message.
            thread_id: Conversation thread identifier for state persistence.

        Returns:
            Final state dictionary containing response, agent_type, tool_name,
            sources, processing_time, and all other state fields.
        """
        from tools.tool_result import ToolResult
        initial_state = {
            "query":               query,
            "chat_history":        "",
            "intent":              "",
            "selected_agent":      "",
            "agent_type":          "",
            "needs_retrieval":     False,
            "retrieved_docs":      [],
            "context":             "",
            "response":            "",
            "sources":             [],
            "confidence":          "",
            "avg_similarity":      0.0,
            "processing_time":     0.0,
            "filter_dict":         None,
            # Phase 7.1 tool fields (kept for compatibility)
            "tool_name":           "",
            "tool_result":         {},
            "calculator_output":   "",
            "doc_analysis_output": "",
            # Phase 7.2 fields
            "tool_result_obj":          ToolResult.skipped(),
            "hybrid_search_metrics":    {},
            "request_log":              {},
            "metrics":             {},
            "error":               "",
        }

        graph_config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(initial_state, config=graph_config)
