from .builder import build_graph

class HealthAssistantWorkflow:
    """
    Wrapper for the LangGraph workflow to easily invoke from app.py.
    """
    def __init__(self, llm, memory, rag):
        self.graph = build_graph(llm, memory, rag)

    def invoke(self, query: str, thread_id: str = "default") -> dict:
        """
        Executes the workflow graph with the given user query.
        Returns the final state dictionary.
        """
        initial_state = {
            "query": query,
            "chat_history": "",
            "intent": "",
            "retrieved_docs": [],
            "context": "",
            "response": "",
            "sources": [],
            "confidence": "",
            "processing_time": 0.0,
            "filter_dict": None,
            "metrics": {},
            "error": ""
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the LangGraph workflow
        final_state = self.graph.invoke(initial_state, config=config)
        return final_state
