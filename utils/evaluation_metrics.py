"""
Evaluation Metrics  –  Phase 7.3
──────────────────────────────────
Reads metrics.json (written by graph/metrics.py after every request) and
computes aggregate statistics for the README performance table and the
'metrics' command in app.py.

Reported metrics:
  - Total requests processed
  - Average retrieval time  (ms)
  - Average LLM time        (s)
  - Average total workflow time (s)
  - Cache hit rate          (%)
  - Documents / Chunks / Agents / Tools counts
  - Duplicate chunk count   (from ChromaDB verify)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


METRICS_FILE = "metrics.json"


@dataclass
class EvaluationSummary:
    """Aggregated performance statistics."""
    total_requests: int          = 0
    avg_retrieval_ms: float      = 0.0
    avg_llm_sec: float           = 0.0
    avg_workflow_sec: float      = 0.0
    cache_hit_rate_pct: float    = 0.0
    document_count: int          = 0
    chunk_count: int             = 0
    agent_count: int             = 6
    tool_count: int              = 4
    duplicate_chunks: int        = 0
    raw_records: List[dict]      = field(default_factory=list)

    def __str__(self) -> str:
        w = 46
        lines = [
            "=" * w,
            "         Evaluation Metrics Summary         ",
            "=" * w,
            f"{'Total Requests':<26}: {self.total_requests}",
            f"{'Documents (MongoDB)':<26}: {self.document_count}",
            f"{'Vector Chunks (ChromaDB)':<26}: {self.chunk_count}",
            f"{'Duplicate Chunks':<26}: {self.duplicate_chunks}",
            f"{'Agents':<26}: {self.agent_count}",
            f"{'Tools':<26}: {self.tool_count}",
            "-" * w,
            f"{'Avg Retrieval Time':<26}: {self.avg_retrieval_ms:.1f} ms",
            f"{'Avg LLM Time':<26}: {self.avg_llm_sec:.2f} s",
            f"{'Avg Workflow Time':<26}: {self.avg_workflow_sec:.2f} s",
            f"{'Cache Hit Rate':<26}: {self.cache_hit_rate_pct:.1f} %",
            "=" * w,
        ]
        return "\n".join(lines)


class EvaluationMetrics:
    """
    Reads metrics.json and computes aggregate statistics.

    Usage:
        em = EvaluationMetrics()
        summary = em.compute(doc_count=51, chunk_count=6061)
        print(summary)
    """

    def __init__(self, metrics_file: str = METRICS_FILE):
        self.metrics_file = metrics_file

    def load_records(self) -> List[Dict[str, Any]]:
        """Load all metric records from the JSON file."""
        if not os.path.exists(self.metrics_file):
            return []
        try:
            with open(self.metrics_file, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def compute(
        self,
        doc_count: int = 0,
        chunk_count: int = 0,
        duplicate_chunks: int = 0,
    ) -> EvaluationSummary:
        """
        Compute aggregate statistics from metrics.json.

        Args:
            doc_count:        Total documents in MongoDB.
            chunk_count:      Total vector chunks in ChromaDB.
            duplicate_chunks: Duplicate chunk count from verify_index.

        Returns:
            EvaluationSummary with all fields populated.
        """
        records = self.load_records()

        if not records:
            return EvaluationSummary(
                document_count   = doc_count,
                chunk_count      = chunk_count,
                duplicate_chunks = duplicate_chunks,
            )

        # ── Retrieval time ────────────────────────────────────────────────────
        retrieval_times = [
            r.get("retriever_time", 0)
            for r in records
            if "retriever_time" in r
        ]
        avg_ret_ms = (
            sum(retrieval_times) / len(retrieval_times)
            if retrieval_times else 0.0
        )

        # ── LLM time ──────────────────────────────────────────────────────────
        llm_times = [
            r.get("llm_time", 0)
            for r in records
            if "llm_time" in r
        ]
        avg_llm = sum(llm_times) / len(llm_times) if llm_times else 0.0

        # ── Total workflow time ───────────────────────────────────────────────
        total_times = [
            r.get("total_time", 0)
            for r in records
            if "total_time" in r
        ]
        avg_total = sum(total_times) / len(total_times) if total_times else 0.0

        # ── Cache hit rate ────────────────────────────────────────────────────
        # Approximated: records without a retriever_time key were cache hits
        cache_hits   = sum(1 for r in records if "retriever_time" not in r)
        cache_rate   = (cache_hits / len(records) * 100) if records else 0.0

        return EvaluationSummary(
            total_requests       = len(records),
            avg_retrieval_ms     = avg_ret_ms,
            avg_llm_sec          = avg_llm,
            avg_workflow_sec     = avg_total,
            cache_hit_rate_pct   = cache_rate,
            document_count       = doc_count,
            chunk_count          = chunk_count,
            agent_count          = 6,
            tool_count           = 4,
            duplicate_chunks     = duplicate_chunks,
            raw_records          = records,
        )

    def reset(self) -> None:
        """Clear metrics.json — useful before a fresh benchmark run."""
        try:
            with open(self.metrics_file, "w") as f:
                json.dump([], f)
        except OSError:
            pass
