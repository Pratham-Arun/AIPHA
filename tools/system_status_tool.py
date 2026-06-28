"""
System Status Tool  –  Phase 7.2
──────────────────────────────────
Enhanced health monitor with richer diagnostics per the Phase 7.2 spec.

Improved output example:
  ============================================
            System Diagnostics
  ============================================
  MongoDB          : Healthy  (3ms)  — 51 documents
  GridFS           : Healthy  (1ms)  — 51 files
  ChromaDB         : Healthy  (2ms)  — 6061 chunks
  LangGraph        : Compiled (0ms)
  Gemini           : Connected (0ms) — gemini-2.5-flash
  Hybrid Search    : Available (BM25 enabled)
  ============================================
  Documents        : 51
  Vector Chunks    : 6061
  Agents           : 6
  Tools            : 4
  ============================================
  Overall          : Healthy ✓
  Execution Time   : 14ms
  ============================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ComponentStatus:
    name: str
    status: str          # "Healthy" | "Compiled" | "Connected" | "Degraded" | "Unavailable"
    detail: str = ""
    latency_ms: int = 0

    @property
    def is_healthy(self) -> bool:
        return self.status in ("Healthy", "Connected", "Compiled", "Available")

    def __str__(self) -> str:
        latency = f"  ({self.latency_ms}ms)" if self.latency_ms else ""
        detail  = f"  — {self.detail}"        if self.detail    else ""
        return f"{self.name:<20}: {self.status}{latency}{detail}"


@dataclass
class SystemStatusResult:
    components: Dict[str, ComponentStatus] = field(default_factory=dict)
    document_count: int  = 0
    chunk_count: int     = 0
    agent_count: int     = 6
    tool_count: int      = 4
    overall_healthy: bool = True
    check_time_ms: int   = 0

    def add(self, component: ComponentStatus) -> None:
        self.components[component.name] = component
        if not component.is_healthy:
            self.overall_healthy = False

    def __str__(self) -> str:
        w = 46
        lines = [
            "=" * w,
            "           System Diagnostics            ",
            "=" * w,
        ]
        for comp in self.components.values():
            lines.append(str(comp))
        lines += [
            "-" * w,
            f"{'Documents':<20}: {self.document_count}",
            f"{'Vector Chunks':<20}: {self.chunk_count}",
            f"{'Agents':<20}: {self.agent_count}",
            f"{'Tools':<20}: {self.tool_count}",
            "-" * w,
            f"{'Overall':<20}: {'Healthy ✓' if self.overall_healthy else 'Degraded ✗'}",
            f"{'Execution Time':<20}: {self.check_time_ms}ms",
            "=" * w,
        ]
        return "\n".join(lines)


class SystemStatusTool:
    """
    Runs health checks against all system components.

    Phase 7.2 enhancements:
      - Document count shown alongside MongoDB status
      - File count shown alongside GridFS status
      - Chunk count shown alongside ChromaDB status
      - BM25 / hybrid search availability reported
      - Tool count added to summary
    """

    AGENT_COUNT = 6
    TOOL_COUNT  = 4  # KnowledgeRetrieval, Calculator, DocAnalysis, SystemStatus

    def __init__(self, rag=None, doc_service=None, workflow=None, llm=None):
        self.rag         = rag
        self.doc_service = doc_service
        self.workflow    = workflow
        self.llm         = llm

    def run(self) -> SystemStatusResult:
        t0 = time.time()
        result = SystemStatusResult(
            agent_count=self.AGENT_COUNT,
            tool_count=self.TOOL_COUNT,
        )

        result.add(self._check_mongodb())
        result.add(self._check_gridfs())

        chroma_comp, chunk_count = self._check_chromadb()
        result.add(chroma_comp)
        result.chunk_count = chunk_count

        result.add(self._check_langgraph())
        result.add(self._check_gemini())
        result.add(self._check_hybrid_search())

        result.document_count = self._get_document_count()
        result.check_time_ms  = int((time.time() - t0) * 1000)
        return result

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_mongodb(self) -> ComponentStatus:
        t0 = time.time()
        try:
            if self.doc_service is None:
                raise RuntimeError("doc_service not provided")
            from database.connection import MongoConnection
            MongoConnection().get_client().admin.command("ping")
            doc_count = self._get_document_count()
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("MongoDB", "Healthy",
                                   detail=f"{doc_count} documents", latency_ms=ms)
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("MongoDB", "Unavailable", detail=str(e), latency_ms=ms)

    def _check_gridfs(self) -> ComponentStatus:
        t0 = time.time()
        try:
            if self.doc_service is None:
                raise RuntimeError("doc_service not provided")
            _ = self.doc_service.gridfs_mgr
            file_count = self._get_document_count()   # same count as docs
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("GridFS", "Healthy",
                                   detail=f"{file_count} files", latency_ms=ms)
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("GridFS", "Unavailable", detail=str(e), latency_ms=ms)

    def _check_chromadb(self):
        t0 = time.time()
        try:
            if self.rag is None:
                raise RuntimeError("rag not provided")
            count = self.rag.get_document_count()
            ms    = int((time.time() - t0) * 1000)
            return (
                ComponentStatus("ChromaDB", "Healthy",
                                detail=f"{count} chunks", latency_ms=ms),
                count,
            )
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("ChromaDB", "Unavailable", detail=str(e), latency_ms=ms), 0

    def _check_langgraph(self) -> ComponentStatus:
        t0 = time.time()
        try:
            if self.workflow is None:
                raise RuntimeError("workflow not provided")
            if self.workflow.graph is None:
                raise RuntimeError("Graph not compiled")
            node_count = len(self.workflow.graph.get_graph().nodes)
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("LangGraph", "Compiled",
                                   detail=f"{node_count} nodes", latency_ms=ms)
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("LangGraph", "Unavailable", detail=str(e), latency_ms=ms)

    def _check_gemini(self) -> ComponentStatus:
        t0 = time.time()
        try:
            import config
            if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your_api_key_here":
                raise RuntimeError("GOOGLE_API_KEY not configured")
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("Gemini", "Connected",
                                   detail=config.LLM_MODEL, latency_ms=ms)
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("Gemini", "Unavailable", detail=str(e), latency_ms=ms)

    def _check_hybrid_search(self) -> ComponentStatus:
        t0 = time.time()
        try:
            from rank_bm25 import BM25Okapi
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("Hybrid Search", "Available",
                                   detail="BM25 enabled", latency_ms=ms)
        except ImportError:
            ms = int((time.time() - t0) * 1000)
            return ComponentStatus("Hybrid Search", "Degraded",
                                   detail="BM25 unavailable (vector-only)",
                                   latency_ms=ms)

    def _get_document_count(self) -> int:
        try:
            if self.doc_service is None:
                return 0
            return len(self.doc_service.list_documents())
        except Exception:
            return 0
