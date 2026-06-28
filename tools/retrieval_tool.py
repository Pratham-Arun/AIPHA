"""
retrieval_tool.py  –  Phase 7.2 shim
──────────────────────────────────────
Re-exports KnowledgeRetrievalTool / KnowledgeRetrievalResult under the
old Phase-7.1 names for full backward compatibility.

All new code should import from tools.knowledge_retrieval_tool directly.
"""
from .knowledge_retrieval_tool import (
    KnowledgeRetrievalTool  as RetrievalTool,
    KnowledgeRetrievalResult as RetrievalResult,
)

__all__ = ["RetrievalTool", "RetrievalResult"]
