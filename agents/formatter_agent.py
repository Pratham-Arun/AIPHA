"""
Response Formatter Agent  –  Phase 7.2
────────────────────────────────────────
Produces consistent, well-structured Markdown output for all agent responses.

Phase 7.2 additions:
  - Hybrid search metrics block (Vector, BM25, Merged, Final, Confidence, Avg Similarity)
  - Tool badge renamed to Knowledge Retrieval Tool where applicable
  - tool_name "knowledge_retrieval_tool" supported
"""

from typing import List, Any, Optional, Dict


AGENT_LABELS = {
    "document_qa":    "📄 Document QA Agent",
    "drug_info":      "💊 Drug Information Agent",
    "nutrition":      "🥗 Nutrition Agent",
    "mental_health":  "🧠 Mental Health Agent",
    "general_health": "💬 General Health Agent",
}

TOOL_LABELS = {
    "knowledge_retrieval_tool":      "🔍 Knowledge Retrieval Tool",
    "retrieval_tool":                "🔍 Knowledge Retrieval Tool",  # backward compat
    "calculator_tool:bmi":           "🧮 Medical Calculator (BMI)",
    "calculator_tool:bmr":           "🧮 Medical Calculator (BMR)",
    "calculator_tool:water_intake":  "🧮 Medical Calculator (Water Intake)",
    "calculator_tool:ideal_weight":  "🧮 Medical Calculator (Ideal Weight)",
    "calculator_tool:bsa":           "🧮 Medical Calculator (BSA)",
    "document_analysis_tool":        "📋 Document Analysis Tool",
}


def _format_citations(sources: List[Any]) -> str:
    if not sources:
        return ""

    seen  = set()
    lines = ["\n\n---\n**📚 Sources:**"]

    for i, (doc, score) in enumerate(sources, 1):
        metadata    = doc.metadata if hasattr(doc, "metadata") else {}
        source_file = metadata.get("source", "Unknown")
        page        = metadata.get("page", "?")
        title       = metadata.get("title", "")
        org         = metadata.get("organization", "")

        if not org or org == "General":
            src_upper = source_file.upper()
            for kw in ["WHO", "CDC", "NIH", "FDA", "AHA"]:
                if kw in src_upper:
                    org = kw
                    break
            else:
                org = "General"

        if not title or title in ("Unknown", "Unknown Document"):
            fname = source_file.replace("\\", "/").split("/")[-1]
            title = fname.rsplit(".", 1)[0] if "." in fname else fname

        confidence = "High" if score < 0.8 else ("Medium" if score < 1.1 else "Low")

        key = f"{title}_{page}"
        if key in seen:
            continue
        seen.add(key)

        lines.append(
            f"{i}. **{org}** — {title} *(Page {page})* · Confidence: {confidence}"
        )

    return "\n".join(lines) if len(lines) > 1 else ""


def _format_hybrid_metrics(
    hybrid_metrics: Dict,
    avg_similarity: float,
    confidence: str,
) -> str:
    """
    Build a compact hybrid search diagnostics block.

    Example:
        🔍 Retrieval Metrics
        Vector Results    : 6
        BM25 Results      : 5
        Merged            : 11
        Duplicates Removed: 3
        Final Results     : 6
        Confidence        : High
        Avg Similarity    : 0.8412
    """
    if not hybrid_metrics:
        return ""

    bm25_used = hybrid_metrics.get("bm25_used", False)
    mode_label = "Hybrid (Vector + BM25)" if bm25_used else "Vector-only"

    lines = [
        "\n\n---",
        f"**🔍 Retrieval Metrics** *(Mode: {mode_label})*",
        f"  Vector Results     : {hybrid_metrics.get('vector_count', 0)}",
    ]
    if bm25_used:
        lines += [
            f"  BM25 Results       : {hybrid_metrics.get('bm25_count', 0)}",
            f"  Merged             : {hybrid_metrics.get('merged_count', 0)}",
            f"  Duplicates Removed : {hybrid_metrics.get('duplicates_removed', 0)}",
        ]
    lines += [
        f"  Final Results      : {hybrid_metrics.get('final_count', 0)}",
        f"  Confidence         : {confidence or 'N/A'}",
    ]
    if avg_similarity:
        lines.append(f"  Avg Similarity     : {avg_similarity:.4f}")
    lines.append(f"  Search Time        : {hybrid_metrics.get('search_time_ms', 0)}ms")

    return "\n".join(lines)


class FormatterAgent:
    """Formats the final response for consistent presentation."""

    def format(
        self,
        response: str,
        agent_type: str,
        sources: Optional[List[Any]] = None,
        processing_time: Optional[float] = None,
        intent: Optional[str] = None,
        tool_name: Optional[str] = None,
        hybrid_search_metrics: Optional[Dict] = None,
        avg_similarity: float = 0.0,
        confidence: str = "",
    ) -> str:
        """
        Format the agent response into the final output string.

        Args:
            response:               Raw response from the specialised agent.
            agent_type:             One of the AGENT_LABELS keys.
            sources:                List of (Document, score) tuples.
            processing_time:        Total workflow execution time in seconds.
            intent:                 Detected intent label.
            tool_name:              Name of tool invoked (if any).
            hybrid_search_metrics:  Dict with vector/BM25/merge counts (Phase 7.2).
            avg_similarity:         Average similarity score across chunks (Phase 7.2).
            confidence:             High / Medium / Low label.

        Returns:
            Fully formatted Markdown response string.
        """
        parts = []

        # 1. Agent label
        agent_label = AGENT_LABELS.get(agent_type, "🤖 AI Health Assistant")
        parts.append(f"*Answered by: {agent_label}*")

        # 2. Tool badge
        if tool_name and tool_name in TOOL_LABELS:
            parts.append(f"*Tool used: {TOOL_LABELS[tool_name]}*")

        parts.append("")  # blank separator line

        # 3. Response body
        parts.append(response)

        # 4. Citations
        if sources:
            citation_block = _format_citations(sources)
            if citation_block:
                parts.append(citation_block)

        # 5. Hybrid search metrics (Phase 7.2)
        if hybrid_search_metrics:
            hm_block = _format_hybrid_metrics(
                hybrid_search_metrics, avg_similarity, confidence
            )
            if hm_block:
                parts.append(hm_block)

        # 6. Processing time footer
        if processing_time is not None:
            parts.append(f"\n\n*⏱ Processing time: {processing_time:.2f}s*")

        return "\n".join(parts)
