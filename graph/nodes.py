"""
Graph Nodes  –  Phase 7.2  (Hybrid Search + Production Improvements)
──────────────────────────────────────────────────────────────────────
Node execution order:

  input_node
    └─> intent_detection_node
          └─> supervisor_node
                ├─> [parallel if needs_retrieval]
                │     memory_node
                │     retriever_node  ← KnowledgeRetrievalTool (Hybrid Search)
                └─> [no retrieval]
                      memory_node
                └─> tool_node         ← Calculator / DocAnalysis / none
                      └─> agent_execution_node
                            └─> formatter_node

Phase 7.2 improvements:
  - retriever_node  backed by KnowledgeRetrievalTool (vector + BM25 hybrid)
  - tool_node       wraps all tool calls in ToolResult for uniform logging
  - formatter_node  includes hybrid search metrics + ToolResult in output
  - All nodes emit structured log lines via the updated logger
  - request_log dict populated end-to-end for diagnostics
"""

import re
import time
from typing import Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .state import HealthAssistantState
from .cache import RetrievalCache
from .logger import (
    log_node_execution, log_error,
    log_tool_execution, log_hybrid_search,
)

from agents.supervisor_agent import SupervisorAgent
from agents.document_agent import DocumentQAAgent
from agents.drug_agent import DrugInformationAgent
from agents.nutrition_agent import NutritionAgent
from agents.mental_health_agent import MentalHealthAgent
from agents.general_health_agent import GeneralHealthAgent
from agents.formatter_agent import FormatterAgent

from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool
from tools.calculator_tool import MedicalCalculatorTool
from tools.document_analysis_tool import DocumentAnalysisTool
from tools.tool_result import ToolResult, measure_ms

# ── Shared retrieval cache (in-memory) ──────────────────────────────────────
_cache = RetrievalCache()


def get_confidence_label(distance: float) -> str:
    if distance < 0.8:
        return "High"
    elif distance < 1.1:
        return "Medium"
    return "Low"


# ── Regex helpers ─────────────────────────────────────────────────────────────
_HEIGHT_RE = re.compile(r"(\d+\.?\d*)\s*(cm|centimeter|metre|meter|m\b)", re.I)
_WEIGHT_RE = re.compile(r"(\d+\.?\d*)\s*(kg|kilogram|lb|pound)", re.I)
_AGE_RE    = re.compile(r"(\d+)\s*(year|yr|age)", re.I)
_GENDER_RE = re.compile(r"\b(male|female|man|woman|boy|girl|m|f)\b", re.I)

_ANALYSIS_TRIGGERS = {
    "analyze", "analyse", "analysis", "summarize", "summarise",
    "summary", "report", "findings", "blood report", "blood test",
    "lab report", "my document", "uploaded",
}


class WorkflowNodes:
    """
    All LangGraph nodes for the Phase 7.2 workflow.
    """

    def __init__(self, llm, memory, rag, doc_service=None):
        self.llm         = llm
        self.memory      = memory
        self.rag         = rag
        self.doc_service = doc_service

        # ── Agents ────────────────────────────────────────────────────────────
        self.supervisor      = SupervisorAgent(llm)
        self.doc_agent       = DocumentQAAgent(llm)
        self.drug_agent      = DrugInformationAgent(llm)
        self.nutrition_agent = NutritionAgent(llm)
        self.mental_agent    = MentalHealthAgent(llm)
        self.general_agent   = GeneralHealthAgent(llm)
        self.formatter       = FormatterAgent()

        # ── Tools ─────────────────────────────────────────────────────────────
        # Phase 7.2: RetrievalTool → KnowledgeRetrievalTool (Hybrid Search)
        self.knowledge_retrieval_tool = KnowledgeRetrievalTool(rag)
        self.calculator_tool          = MedicalCalculatorTool()
        self.doc_analysis_tool        = (
            DocumentAnalysisTool(doc_service) if doc_service else None
        )

    # ────────────────────────────────────────────────────────────────────────
    # 1. Input Node
    # ────────────────────────────────────────────────────────────────────────
    def input_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        t0 = time.time()
        try:
            query       = state["query"]
            query_lower = query.lower()
            filter_dict = None
            if "who" in query_lower or "world health organization" in query_lower:
                filter_dict = {"source": {"$contains": "WHO"}}
            elif "cdc" in query_lower:
                filter_dict = {"source": {"$contains": "CDC"}}
            elif "nih" in query_lower:
                filter_dict = {"source": {"$contains": "NIH"}}

            log_node_execution("input_node", time.time() - t0, "Success")
            return {
                "filter_dict":          filter_dict,
                "metrics":              {"start_time": t0},
                "error":                "",
                # Phase 7.1 compat
                "tool_name":            "",
                "tool_result":          {},
                "calculator_output":    "",
                "doc_analysis_output":  "",
                # Phase 7.2
                "tool_result_obj":      ToolResult.skipped(),
                "hybrid_search_metrics": {},
                "request_log":          {"query": query[:80]},
                "avg_similarity":       0.0,
            }
        except Exception as e:
            log_error("input_node", e)
            return {"error": str(e), "metrics": {"start_time": time.time()}}

    # ────────────────────────────────────────────────────────────────────────
    # 2. Intent Detection Node
    # ────────────────────────────────────────────────────────────────────────
    def intent_detection_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        t0 = time.time()
        try:
            prompt = ChatPromptTemplate.from_template(
                "Classify the following user message into one of these intents: "
                "Greeting, General Chat, Medical Question, Nutrition, Mental Health, "
                "Medicine Information, Medical Report Analysis, Emergency, Follow-up Question.\n"
                "Respond ONLY with the intent name.\n\nMessage: {query}"
            )
            chain  = prompt | self.llm | StrOutputParser()
            intent = chain.invoke({"query": state["query"]}).strip()
            intent = intent.replace(".", "").replace('"', "").replace("'", "").strip()

            valid = [
                "Greeting", "General Chat", "Medical Question", "Nutrition",
                "Mental Health", "Medicine Information", "Medical Report Analysis",
                "Emergency", "Follow-up Question",
            ]
            if intent not in valid:
                intent = "General Chat"

            elapsed = time.time() - t0
            log_node_execution("intent_detection_node", elapsed, "Success", f"Intent: {intent}")
            return {
                "intent": intent,
                "request_log": {"intent": intent},
            }
        except Exception as e:
            log_error("intent_detection_node", e)
            return {"intent": "General Chat", "error": str(e)}

    # ────────────────────────────────────────────────────────────────────────
    # 3. Supervisor Node
    # ────────────────────────────────────────────────────────────────────────
    def supervisor_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        t0 = time.time()
        try:
            query  = state["query"]
            intent = state.get("intent", "General Chat")

            selected_agent  = self.supervisor.select_agent(query, intent)
            needs_retrieval = self.supervisor.requires_retrieval(selected_agent)

            elapsed = time.time() - t0
            log_node_execution(
                "supervisor_node", elapsed, "Success",
                f"Agent: {selected_agent} | Retrieval: {needs_retrieval}"
            )
            return {
                "selected_agent":  selected_agent,
                "agent_type":      selected_agent,
                "needs_retrieval": needs_retrieval,
                "request_log":     {"agent": selected_agent},
            }
        except Exception as e:
            log_error("supervisor_node", e)
            return {
                "selected_agent":  "document_qa",
                "agent_type":      "document_qa",
                "needs_retrieval": True,
                "error":           str(e),
            }

    # ────────────────────────────────────────────────────────────────────────
    # 4. Memory Node
    # ────────────────────────────────────────────────────────────────────────
    def memory_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        t0 = time.time()
        try:
            history = self.memory.get_formatted_history()
            if len(history) > 2000:
                history = history[-2000:]
            log_node_execution("memory_node", time.time() - t0, "Success")
            return {"chat_history": history}
        except Exception as e:
            log_error("memory_node", e)
            return {"chat_history": "", "error": str(e)}

    # ────────────────────────────────────────────────────────────────────────
    # 5. Retriever Node  –  Phase 7.2: KnowledgeRetrievalTool (Hybrid Search)
    # ────────────────────────────────────────────────────────────────────────
    def retriever_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """
        Retrieves relevant document chunks via Hybrid Search.
        Results are cached; cache is keyed on (query, filter_dict).
        """
        t0 = time.time()
        try:
            query       = state["query"]
            filter_dict = state.get("filter_dict")

            # ── Cache check ────────────────────────────────────────────────
            cached = _cache.get(query, filter_dict)
            if cached:
                context, results, discarded, search_ms, hsm = cached
                log_node_execution("retriever_node", time.time() - t0, "Cache Hit")
                hybrid_metrics = hsm
            else:
                # ── Hybrid Search ──────────────────────────────────────────
                kr = self.knowledge_retrieval_tool.run(query, filter_dict=filter_dict)

                context   = kr.context
                results   = kr.chunks
                discarded = kr.discarded_count
                search_ms = kr.search_time_ms
                hybrid_metrics = {
                    "vector_count":      kr.vector_count,
                    "bm25_count":        kr.bm25_count,
                    "merged_count":      kr.merged_count,
                    "duplicates_removed":kr.duplicates_removed,
                    "final_count":       kr.final_count,
                    "bm25_used":         kr.bm25_used,
                    "search_time_ms":    search_ms,
                }
                _cache.set(
                    query,
                    (context, results, discarded, search_ms, hybrid_metrics),
                    filter_dict,
                )
                log_hybrid_search(kr)
                log_node_execution(
                    "retriever_node", time.time() - t0, "Success",
                    kr.format_summary()
                )

            sources          = [(doc, score) for doc, score in results]
            confidences      = [score for _, score in results]
            confidence_label = get_confidence_label(min(confidences)) if confidences else "Low"
            avg_sim          = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

            return {
                "retrieved_docs":        results,
                "context":               context,
                "sources":               sources,
                "confidence":            confidence_label,
                "avg_similarity":        avg_sim,
                "hybrid_search_metrics": hybrid_metrics,
                "metrics":               {"retriever_time": search_ms},
                # Phase 7.1 compat field – still used by formatter
                "tool_name":             "knowledge_retrieval_tool",
                "request_log":           {"retrieval_ms": search_ms},
            }
        except Exception as e:
            log_error("retriever_node", e)
            return {
                "retrieved_docs": [], "context": "", "sources": [],
                "confidence": "Low", "avg_similarity": 0.0,
                "hybrid_search_metrics": {}, "error": str(e),
            }

    # ────────────────────────────────────────────────────────────────────────
    # 6. Tool Node  –  Phase 7.2: ToolResult wrapper around every tool call
    # ────────────────────────────────────────────────────────────────────────
    def tool_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        """
        Deterministic tool dispatch. Wraps all tool calls in ToolResult.
        """
        t0         = time.time()
        agent_type = state.get("selected_agent", "general_health")
        query      = state.get("query", "")
        query_lower = query.lower()

        base: Dict[str, Any] = {
            "tool_name":           "",
            "tool_result":         {},
            "calculator_output":   "",
            "doc_analysis_output": "",
        }

        # ── Nutrition → MedicalCalculatorTool ────────────────────────────────
        if agent_type == "nutrition":
            with measure_ms() as t:
                calc_outcome = self._try_calculator(query, query_lower)
            if calc_outcome:
                tool_name, output_str, result_dict = calc_outcome
                tr = ToolResult.success(
                    tool_name         = tool_name,
                    output            = output_str,
                    result            = result_dict,
                    execution_time_ms = t.elapsed_ms,
                    input_summary     = query[:60],
                )
                log_tool_execution(tr)
                log_node_execution("tool_node", time.time() - t0, "Calculator", tr.log_line())
                return {
                    **base,
                    "tool_name":         tool_name,
                    "tool_result":       result_dict,
                    "calculator_output": output_str,
                    "tool_result_obj":   tr,
                    "request_log":       {"tool": tool_name, "tool_time_ms": t.elapsed_ms},
                }

        # ── Document QA → DocumentAnalysisTool ───────────────────────────────
        if agent_type == "document_qa" and self.doc_analysis_tool:
            if any(kw in query_lower for kw in _ANALYSIS_TRIGGERS):
                with measure_ms() as t:
                    analysis = self._try_doc_analysis(query_lower)
                if analysis:
                    tr = ToolResult.success(
                        tool_name         = "document_analysis_tool",
                        output            = str(analysis),
                        result            = analysis.to_dict(),
                        execution_time_ms = t.elapsed_ms,
                        confidence        = analysis.doc_type_confidence,
                        input_summary     = query[:60],
                    )
                    log_tool_execution(tr)
                    log_node_execution("tool_node", time.time() - t0, "DocAnalysis", tr.log_line())
                    return {
                        **base,
                        "tool_name":            "document_analysis_tool",
                        "tool_result":          analysis.to_dict(),
                        "doc_analysis_output":  str(analysis),
                        "tool_result_obj":      tr,
                        "request_log": {
                            "tool":         "document_analysis_tool",
                            "tool_time_ms": t.elapsed_ms,
                        },
                    }

        # ── No tool ───────────────────────────────────────────────────────────
        tr = ToolResult.skipped(f"No tool required for agent: {agent_type}")
        log_node_execution("tool_node", time.time() - t0, "Skipped", tr.log_line())
        return {
            **base,
            "tool_result_obj": tr,
            "request_log":     {"tool": "none"},
        }

    # ────────────────────────────────────────────────────────────────────────
    # 7. Agent Execution Node
    # ────────────────────────────────────────────────────────────────────────
    def agent_execution_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        t0 = time.time()
        try:
            query               = state["query"]
            agent_type          = state.get("selected_agent", "general_health")
            context             = state.get("context", "")
            chat_history        = state.get("chat_history", "")
            intent              = state.get("intent", "")
            calculator_output   = state.get("calculator_output", "")
            doc_analysis_output = state.get("doc_analysis_output", "")

            if agent_type == "document_qa":
                response = self.doc_agent.run(
                    query, context, chat_history,
                    doc_analysis_output=doc_analysis_output
                )
            elif agent_type == "drug_info":
                response = self.drug_agent.run(query, context, chat_history)
            elif agent_type == "nutrition":
                response = self.nutrition_agent.run(
                    query, context, chat_history,
                    calculator_output=calculator_output
                )
            elif agent_type == "mental_health":
                response = self.mental_agent.run(query, context, chat_history)
            else:
                prefix = ""
                if intent == "Emergency":
                    prefix = (
                        "**🚨 EMERGENCY NOTICE:** Please contact emergency services "
                        "immediately (dial **911** or go to the nearest ER). "
                        "I am an AI — not a doctor.\n\n"
                    )
                response = prefix + self.general_agent.run(query, chat_history)

            llm_ms = int((time.time() - t0) * 1000)
            log_node_execution(
                "agent_execution_node", time.time() - t0,
                "Success", f"Agent: {agent_type} | LLM: {llm_ms}ms"
            )
            return {
                "response":    response,
                "metrics":     {"llm_time": llm_ms / 1000},
                "request_log": {"llm_ms": llm_ms},
            }
        except Exception as e:
            log_error("agent_execution_node", e)
            return {
                "response": "I'm sorry, I encountered an error processing your request.",
                "error":    str(e),
            }

    # ────────────────────────────────────────────────────────────────────────
    # 8. Formatter Node  –  Phase 7.2: hybrid metrics + ToolResult in output
    # ────────────────────────────────────────────────────────────────────────
    def formatter_node(self, state: HealthAssistantState) -> Dict[str, Any]:
        t0 = time.time()
        try:
            metrics    = state.get("metrics", {})
            total_time = time.time() - metrics.get("start_time", t0)
            metrics["total_time"] = total_time

            from .metrics import export_metrics
            # Merge ToolResult dict into metrics for export
            tr_obj = state.get("tool_result_obj")
            if tr_obj:
                metrics["tool"] = tr_obj.to_dict()
            export_metrics(metrics)

            formatted = self.formatter.format(
                response              = state.get("response", ""),
                agent_type            = state.get("agent_type", "general_health"),
                sources               = state.get("sources", []),
                processing_time       = total_time,
                intent                = state.get("intent"),
                tool_name             = state.get("tool_name", ""),
                hybrid_search_metrics = state.get("hybrid_search_metrics", {}),
                avg_similarity        = state.get("avg_similarity", 0.0),
                confidence            = state.get("confidence", ""),
            )

            # Emit full request log
            rlog = state.get("request_log", {})
            rlog.update({
                "total_ms":       int(total_time * 1000),
                "retrieval_ms":   int(metrics.get("retriever_time", 0)),
                "llm_ms":         int(metrics.get("llm_time", 0) * 1000),
                "status":         "ok" if not state.get("error") else "error",
            })

            log_node_execution(
                "formatter_node", time.time() - t0,
                "Success", f"Total: {total_time:.2f}s"
            )
            return {
                "response":        formatted,
                "processing_time": total_time,
                "request_log":     rlog,
            }
        except Exception as e:
            log_error("formatter_node", e)
            return {"error": str(e)}

    # ────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────────────

    def _try_calculator(self, query: str, query_lower: str):
        """Returns (tool_name, output_str, result_dict) or None."""
        calc_type = self._detect_calc_type(query_lower)
        if not calc_type:
            return None

        height   = self._extract_value(_HEIGHT_RE, query)
        weight   = self._extract_value(_WEIGHT_RE, query)
        age_m    = _AGE_RE.search(query)
        gender_m = _GENDER_RE.search(query)
        age      = int(age_m.group(1))   if age_m    else None
        gender   = gender_m.group(1)     if gender_m else None

        try:
            if calc_type == "bmi" and height and weight:
                r = self.calculator_tool.bmi(height_cm=height, weight_kg=weight)
                return "calculator_tool:bmi", str(r), r.to_dict()
            elif calc_type == "bmr" and height and weight and age and gender:
                r = self.calculator_tool.bmr(age=age, height_cm=height,
                                             weight_kg=weight, gender=gender)
                return "calculator_tool:bmr", str(r), r.to_dict()
            elif calc_type == "water_intake" and weight:
                r = self.calculator_tool.water_intake(weight_kg=weight)
                return "calculator_tool:water_intake", str(r), r.to_dict()
            elif calc_type == "ideal_weight" and height and gender:
                r = self.calculator_tool.ideal_body_weight(height_cm=height, gender=gender)
                return "calculator_tool:ideal_weight", str(r), r.to_dict()
        except ValueError:
            pass
        return None

    def _try_doc_analysis(self, query_lower: str):
        """Returns DocumentAnalysisResult or None."""
        if not self.doc_analysis_tool:
            return None
        try:
            docs = self.doc_analysis_tool.list_available()
            if not docs:
                return None
            chosen = next(
                (d for d in docs
                 if any(w in query_lower for w in d["title"].lower().split() if len(w) > 3)),
                docs[0]
            )
            result = self.doc_analysis_tool.run_by_id(chosen["doc_id"])
            return result if result.success else None
        except Exception:
            return None

    @staticmethod
    def _detect_calc_type(query_lower: str):
        if any(t in query_lower for t in ("bmi", "body mass index")):
            return "bmi"
        if any(t in query_lower for t in ("bmr", "basal metabolic", "calorie", "calories")):
            return "bmr"
        if any(t in query_lower for t in ("water intake", "hydration", "water")):
            return "water_intake"
        if any(t in query_lower for t in ("ideal weight", "ideal body weight", "ibw")):
            return "ideal_weight"
        if _HEIGHT_RE.search(query_lower) and _WEIGHT_RE.search(query_lower):
            return "bmi"
        return None

    @staticmethod
    def _extract_value(pattern, text: str):
        match = pattern.search(text)
        if match:
            raw  = float(match.group(1))
            unit = match.group(2).lower() if match.lastindex >= 2 else ""
            if unit in ("lb", "pound"):
                raw *= 0.453592
            return raw
        return None

    # ── Legacy stubs ─────────────────────────────────────────────────────────
    def prompt_builder_node(self, state):   return {}
    def llm_node(self, state):              return self.agent_execution_node(state)
    def general_response_node(self, state): return self.agent_execution_node(state)
    def citation_formatter_node(self, state): return self.formatter_node(state)
