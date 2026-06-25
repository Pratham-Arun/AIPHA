import time
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import HealthAssistantState
from prompts import health_assistant_prompt
from .cache import RetrievalCache
from .logger import log_node_execution, log_error

# Global retrieval cache
_cache = RetrievalCache()

def get_confidence_label(distance: float) -> str:
    """Map a ChromaDB distance score to a human-readable confidence label."""
    if distance < 0.8:
        return "High"
    elif distance < 1.1:
        return "Medium"
    else:
        return "Low"

class WorkflowNodes:
    """
    Contains all nodes for the LangGraph workflow.
    Requires initialized LLM, MemoryManager, and RAGPipeline instances.
    """
    def __init__(self, llm, memory, rag):
        self.llm = llm
        self.memory = memory
        self.rag = rag

    def input_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """Initializes processing and parses basic query filters if needed."""
        start_time = time.time()
        try:
            query = state["query"]
            filter_dict = None
            input_lower = query.lower()
            if "who" in input_lower or "world health organization" in input_lower:
                filter_dict = {"source": {"$contains": "WHO"}}
            elif "cdc" in input_lower:
                filter_dict = {"source": {"$contains": "CDC"}}
            elif "nih" in input_lower:
                filter_dict = {"source": {"$contains": "NIH"}}
                
            metrics = {"start_time": start_time}
            log_node_execution("input_node", time.time() - start_time, "Success")
            return {"filter_dict": filter_dict, "metrics": metrics, "error": ""}
        except Exception as e:
            log_error("input_node", e)
            return {"error": str(e)}

    def memory_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """Fetches memory from MemoryManager."""
        start_time = time.time()
        try:
            # Memory Compression: If history is too long, MemoryManager should summarize it.
            # Assuming MemoryManager handles its own string length.
            history = self.memory.get_formatted_history()
            
            # Simple compression logic
            if len(history) > 2000:
                history = history[-2000:]
            
            log_node_execution("memory_node", time.time() - start_time, "Success")
            return {"chat_history": history}
        except Exception as e:
            log_error("memory_node", e)
            return {"chat_history": "", "error": str(e)}

    def intent_detection_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """Detects the intent of the user query."""
        start_time = time.time()
        try:
            prompt = ChatPromptTemplate.from_template(
                "Classify the following user message into one of these intents: "
                "Greeting, General Chat, Medical Question, Nutrition, Mental Health, "
                "Medicine Information, Medical Report Analysis, Emergency, Follow-up Question.\n"
                "Respond ONLY with the intent name.\n\nMessage: {query}"
            )
            chain = prompt | self.llm | StrOutputParser()
            intent = chain.invoke({"query": state["query"]}).strip()
            
            # Cleanup potential extra punctuation or whitespace
            intent = intent.replace(".", "").replace('"', '').replace("'", "")
            
            valid_intents = [
                "Greeting", "General Chat", "Medical Question", "Nutrition", 
                "Mental Health", "Medicine Information", "Medical Report Analysis", 
                "Emergency", "Follow-up Question"
            ]
            if intent not in valid_intents:
                intent = "General Chat"
                    
            log_node_execution("intent_detection_node", time.time() - start_time, "Success", f"Intent: {intent}")
            return {"intent": intent}
        except Exception as e:
            log_error("intent_detection_node", e)
            return {"intent": "General Chat", "error": str(e)}

    def retriever_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """Retrieves documents from RAG with caching."""
        start_time = time.time()
        try:
            query = state["query"]
            filter_dict = state.get("filter_dict")
            
            # Check cache
            cached_result = _cache.get(query, filter_dict)
            if cached_result:
                context, results, discarded_count, search_time_ms = cached_result
                log_node_execution("retriever_node", time.time() - start_time, "Cache Hit")
            else:
                context, results, discarded_count, search_time_ms = self.rag.retrieve(query, filter_dict=filter_dict)
                _cache.set(query, (context, results, discarded_count, search_time_ms), filter_dict)
                log_node_execution("retriever_node", time.time() - start_time, "Success", f"Retrieved {len(results)} chunks")
            
            sources = []
            confidences = []
            for doc, score in results:
                sources.append((doc, score))
                confidences.append(score)
                
            confidence_label = "Low"
            if confidences:
                confidence_label = get_confidence_label(min(confidences))
                
            return {
                "retrieved_docs": results,
                "context": context,
                "sources": sources,
                "confidence": confidence_label,
                "metrics": {"retriever_time": search_time_ms}
            }
        except Exception as e:
            log_error("retriever_node", e)
            return {"retrieved_docs": [], "context": "", "sources": [], "confidence": "Low", "error": str(e)}

    def prompt_builder_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """Combines state variables to build the prompt for LLM."""
        start_time = time.time()
        try:
            context = state.get("context", "")
            log_node_execution("prompt_builder_node", time.time() - start_time, "Success")
            return {"context": context}
        except Exception as e:
            log_error("prompt_builder_node", e)
            return {"error": str(e)}

    def llm_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """Invokes the Gemini LLM."""
        start_time = time.time()
        try:
            context = state.get("context", "")
            chain = health_assistant_prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "question": state["query"],
                "chat_history": state.get("chat_history", ""),
                "context": context
            })
            
            if state.get("intent") == "Emergency":
                response = "**EMERGENCY NOTICE:** Please contact emergency services immediately (e.g., dial 911 or visit the nearest ER). I am an AI, not a doctor.\n\n" + response
                
            llm_time = time.time() - start_time
            log_node_execution("llm_node", llm_time, "Success")
            return {"response": response, "metrics": {"llm_time": llm_time}}
        except Exception as e:
            log_error("llm_node", e)
            return {"response": "I'm sorry, I encountered an error while processing your request.", "error": str(e)}

    def general_response_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """Provides a general response when confidence is low."""
        start_time = time.time()
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are an AI Health Assistant.\n"
                "The user asked: {question}\n"
                "Please answer using your general knowledge, but include a disclaimer that this information is not from the provided medical documents and they should consult a professional."
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": state["query"]})
            llm_time = time.time() - start_time
            log_node_execution("general_response_node", llm_time, "Success")
            return {"response": response, "metrics": {"llm_time": llm_time}}
        except Exception as e:
            log_error("general_response_node", e)
            return {"response": "I'm sorry, I encountered an error while processing your request.", "error": str(e)}

    def citation_formatter_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """Formats the final response and collects metrics."""
        start_time = time.time()
        try:
            metrics = state.get("metrics", {})
            total_time = time.time() - metrics.get("start_time", start_time)
            metrics["total_time"] = total_time
            
            from .metrics import export_metrics
            export_metrics(metrics)
            
            log_node_execution("citation_formatter_node", time.time() - start_time, "Success", f"Total Workflow Time: {total_time:.2f}s")
            return {"processing_time": total_time}
        except Exception as e:
            log_error("citation_formatter_node", e)
            return {"error": str(e)}
