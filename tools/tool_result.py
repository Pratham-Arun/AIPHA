"""
ToolResult  –  Phase 7.2
──────────────────────────
Standardised envelope for every tool execution in the system.

Replaces the ad-hoc trio of:
  - calculator_output   (str)
  - doc_analysis_output (str)
  - tool_result         (dict)

with a single, consistent object that:
  - carries the tool name and status
  - records execution time
  - holds a confidence label
  - stores the formatted output string for LLM injection
  - stores the raw structured result dict for downstream processing / logging
  - captures failure reasons without crashing the workflow

Design:
  Every tool wraps its output in a ToolResult before returning.
  The tool_node stores a single ToolResult in state["tool_result_obj"].
  The agent_execution_node and formatter_node read from it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """
    Standardised result envelope for any tool execution.

    Attributes:
        tool_name       Name of the tool that produced this result.
                        e.g. "knowledge_retrieval_tool", "calculator_tool:bmi"
        status          "success" | "skipped" | "error"
        execution_time_ms  Wall-clock time for the tool call in milliseconds.
        confidence      "High" | "Medium" | "Low" | "N/A"
        output          Human-readable formatted string ready for LLM injection.
        result          Serialisable dict with raw structured data.
        error_reason    Populated when status == "error".
        input_summary   Short description of the inputs used.
    """

    tool_name: str                        = "none"
    status: str                           = "skipped"   # success | skipped | error
    execution_time_ms: int                = 0
    confidence: str                       = "N/A"
    output: str                           = ""          # formatted for LLM prompt
    result: Dict[str, Any]                = field(default_factory=dict)
    error_reason: str                     = ""
    input_summary: str                    = ""

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def success(
        cls,
        tool_name: str,
        output: str,
        result: Dict[str, Any],
        execution_time_ms: int,
        confidence: str = "N/A",
        input_summary: str = "",
    ) -> "ToolResult":
        """Create a successful ToolResult."""
        return cls(
            tool_name         = tool_name,
            status            = "success",
            execution_time_ms = execution_time_ms,
            confidence        = confidence,
            output            = output,
            result            = result,
            input_summary     = input_summary,
        )

    @classmethod
    def skipped(cls, reason: str = "") -> "ToolResult":
        """Create a ToolResult for when no tool was needed."""
        return cls(
            tool_name    = "none",
            status       = "skipped",
            error_reason = reason,
        )

    @classmethod
    def error(
        cls,
        tool_name: str,
        error_reason: str,
        execution_time_ms: int = 0,
    ) -> "ToolResult":
        """Create a ToolResult for a failed tool invocation."""
        return cls(
            tool_name         = tool_name,
            status            = "error",
            execution_time_ms = execution_time_ms,
            error_reason      = error_reason,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def was_skipped(self) -> bool:
        return self.status == "skipped"

    # ── Logging helpers ───────────────────────────────────────────────────────

    def log_line(self) -> str:
        """
        One-line summary for structured logging.

        Example:
            "Tool: calculator_tool:bmi | Status: success | Time: 2ms | Confidence: N/A"
        """
        parts = [
            f"Tool: {self.tool_name}",
            f"Status: {self.status}",
            f"Time: {self.execution_time_ms}ms",
        ]
        if self.confidence != "N/A":
            parts.append(f"Confidence: {self.confidence}")
        if self.input_summary:
            parts.append(f"Input: {self.input_summary}")
        if self.error_reason:
            parts.append(f"Error: {self.error_reason}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable representation suitable for metrics export."""
        return {
            "tool_name":          self.tool_name,
            "status":             self.status,
            "execution_time_ms":  self.execution_time_ms,
            "confidence":         self.confidence,
            "output_length":      len(self.output),
            "error_reason":       self.error_reason,
            "input_summary":      self.input_summary,
        }

    def __str__(self) -> str:
        return self.log_line()


# ── Timer context-manager helper (used inside tool wrappers) ──────────────────

class _Timer:
    """Simple elapsed-ms timer."""
    def __enter__(self):
        self._start = time.time()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = int((time.time() - self._start) * 1000)

def measure_ms() -> _Timer:
    """Usage:  with measure_ms() as t: ...; print(t.elapsed_ms)"""
    return _Timer()
