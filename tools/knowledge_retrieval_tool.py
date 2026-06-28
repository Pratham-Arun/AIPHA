"""
Knowledge Retrieval Tool  –  Phase 7.2
────────────────────────────────────────
Renamed and enhanced version of the Phase 7.1 RetrievalTool.

Combines:
  1. Semantic vector search (ChromaDB / HuggingFace embeddings)
  2. BM25 keyword search   (rank_bm25 – fully local, no API needed)

into a single Hybrid Search result, giving better recall for:
  - Exact medical abbreviations (HbA1c, WBC, TSH …)
  - Organization names (WHO, CDC, NIH …)
  - Drug names (Metformin, Aspirin …)
  - Queries where semantic similarity alone is insufficient

Architecture:
  User Query
      │
      ├─► Vector Search (ChromaDB)   ─► top-K semantic chunks
      └─► BM25 Search   (rank_bm25)  ─► top-K keyword chunks
              │
              ▼
       Merge & Deduplicate
              │
              ▼
       Reciprocal Rank Fusion  (RRF) scoring
              │
              ▼
       Final Ranked Results  →  LLM

Naming rationale:
  "Knowledge Retrieval Tool" is used instead of "Retrieval Tool" because
  future versions may add web search, PubMed, or medical database backends
  behind the same interface.

Used by:
  - Document QA Agent
  - Drug Information Agent
  - Nutrition Agent
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Dict

import config
from rag.retriever import format_retrieved_docs

# BM25 import — gracefully degraded to vector-only if unavailable
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class KnowledgeRetrievalResult:
    """
    Structured result from the Knowledge Retrieval Tool.

    Extends the Phase 7.1 RetrievalResult with hybrid-search metrics.
    """

    # Pre-formatted string ready to inject into the LLM prompt
    context: str

    # Final merged (Document, score) pairs after RRF ranking
    chunks: List[Tuple[Any, float]] = field(default_factory=list)

    # ── Hybrid search metrics ────────────────────────────────────────────────
    vector_count: int = 0          # chunks from ChromaDB vector search
    bm25_count: int = 0            # chunks from BM25 keyword search
    merged_count: int = 0          # total after merge (before dedup)
    duplicates_removed: int = 0    # exact-content duplicates removed
    final_count: int = 0           # chunks after dedup → sent to LLM

    # Number of chunks discarded by the distance threshold (vector path)
    discarded_count: int = 0

    # Total search time across both backends
    search_time_ms: int = 0

    # Human-readable confidence derived from the best score
    confidence: str = "Low"

    # Average similarity score (for reporting)
    avg_similarity: float = 0.0

    # Whether BM25 was actually used (depends on corpus being loaded)
    bm25_used: bool = False

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0

    @staticmethod
    def confidence_label(score: float) -> str:
        """Classify a distance score as High / Medium / Low."""
        if score < 0.8:
            return "High"
        elif score < 1.1:
            return "Medium"
        return "Low"

    def __post_init__(self) -> None:
        self.final_count = len(self.chunks)
        if self.chunks:
            scores = [s for _, s in self.chunks]
            best   = min(scores)
            self.confidence    = self.confidence_label(best)
            self.avg_similarity = round(sum(scores) / len(scores), 4)

    def format_summary(self) -> str:
        """One-line summary for console / log output."""
        if self.bm25_used:
            return (
                f"Knowledge Retrieval Tool (Hybrid): "
                f"Vector={self.vector_count} | BM25={self.bm25_count} | "
                f"Merged={self.merged_count} | Dedup={self.duplicates_removed} | "
                f"Final={self.final_count} | "
                f"Confidence={self.confidence} | Avg Similarity={self.avg_similarity} | "
                f"Search={self.search_time_ms}ms"
            )
        return (
            f"Knowledge Retrieval Tool (Vector-only): "
            f"Chunks={self.final_count} | "
            f"Confidence={self.confidence} | Avg Similarity={self.avg_similarity} | "
            f"Search={self.search_time_ms}ms"
        )


# ── RRF scoring constant ──────────────────────────────────────────────────────
_RRF_K = 60   # standard RRF constant; higher → more equalisation between lists


class KnowledgeRetrievalTool:
    """
    Hybrid semantic + BM25 retrieval tool.

    Usage:
        tool = KnowledgeRetrievalTool(rag_pipeline)
        result = tool.run("What causes hypertension?")
        print(result.context)           # formatted text ready for LLM
        print(result.format_summary())  # one-line diagnostic log
    """

    def __init__(self, rag_pipeline):
        """
        Args:
            rag_pipeline: Initialised RAGPipeline instance.
        """
        self.rag       = rag_pipeline
        self.top_k     = config.RETRIEVER_TOP_K
        self.threshold = config.RETRIEVER_DISTANCE_THRESHOLD

        # BM25 corpus — built lazily on first search
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: List[Tuple[Any, float]] = []  # (Document, vector_score)
        self._corpus_built: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        filter_dict: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> KnowledgeRetrievalResult:
        """
        Perform hybrid (vector + BM25) retrieval for *query*.

        Falls back to vector-only if BM25 is unavailable or corpus is empty.

        Args:
            query:       User query string.
            filter_dict: Optional ChromaDB metadata filter.
            top_k:       Override the default number of final chunks.

        Returns:
            KnowledgeRetrievalResult with merged and ranked chunks.
        """
        effective_k = top_k or self.top_k
        t0 = time.time()

        # ── 1. Vector Search ────────────────────────────────────────────────
        vector_results, vec_discarded, vec_ms = self._vector_search(
            query, filter_dict, effective_k
        )

        # ── 2. BM25 Search ──────────────────────────────────────────────────
        bm25_results, bm25_used = [], False
        if _BM25_AVAILABLE and vector_results:
            bm25_results, bm25_used = self._bm25_search(
                query, vector_results, effective_k
            )

        # ── 3. Merge + Deduplicate + RRF Rank ───────────────────────────────
        merged, duplicates = self._merge(vector_results, bm25_results, effective_k)

        # ── 4. Build context string ─────────────────────────────────────────
        context = format_retrieved_docs(merged) if merged else "No relevant medical documents found."

        total_ms = int((time.time() - t0) * 1000)

        return KnowledgeRetrievalResult(
            context          = context,
            chunks           = merged,
            vector_count     = len(vector_results),
            bm25_count       = len(bm25_results),
            merged_count     = len(vector_results) + len(bm25_results),
            duplicates_removed = duplicates,
            discarded_count  = vec_discarded,
            search_time_ms   = total_ms,
            bm25_used        = bm25_used,
        )

    def invalidate_bm25_cache(self) -> None:
        """
        Force rebuild of the BM25 index on next search.
        Call this after new documents are indexed.
        """
        self._bm25_index  = None
        self._bm25_corpus = []
        self._corpus_built = False

    # ── Private: Vector Search ────────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        filter_dict: Optional[dict],
        top_k: int,
    ) -> Tuple[List[Tuple[Any, float]], int, int]:
        """
        Returns (relevant_results, discarded_count, search_time_ms).
        """
        try:
            context_str, results, discarded, ms = self.rag.retrieve(
                query, filter_dict=filter_dict
            )
            return results, discarded, ms
        except Exception:
            return [], 0, 0

    # ── Private: BM25 Search ─────────────────────────────────────────────────

    def _build_bm25_index(self, corpus: List[Tuple[Any, float]]) -> None:
        """
        Build a BM25Okapi index from a list of (Document, score) pairs.
        Tokenises each document's page_content into lowercase words.
        """
        tokenised = [
            self._tokenise(doc.page_content)
            for doc, _ in corpus
        ]
        non_empty = [t for t in tokenised if t]
        if not non_empty:
            return
        self._bm25_index  = BM25Okapi(tokenised)
        self._bm25_corpus = corpus
        self._corpus_built = True

    def _bm25_search(
        self,
        query: str,
        vector_results: List[Tuple[Any, float]],
        top_k: int,
    ) -> Tuple[List[Tuple[Any, float]], bool]:
        """
        Re-rank and supplement the vector results using BM25 scores.

        Strategy:
          - Uses the vector result set as the BM25 corpus (efficient — no full-
            collection scan). This gives BM25 a chance to re-rank within the
            top-K semantic results based on keyword overlap.
          - Returns the top-K re-ranked results.

        Returns:
            (ranked_results, bm25_was_used)
        """
        if not _BM25_AVAILABLE or not vector_results:
            return [], False

        try:
            # Rebuild index if corpus changed
            if not self._corpus_built or len(self._bm25_corpus) != len(vector_results):
                self._build_bm25_index(vector_results)

            if self._bm25_index is None:
                return [], False

            tokens  = self._tokenise(query)
            scores  = self._bm25_index.get_scores(tokens)

            # Pair each document with its BM25 score
            ranked = sorted(
                zip(self._bm25_corpus, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            # Convert BM25 score to a pseudo-distance (lower = better, like ChromaDB)
            # Normalise: max_score → 0.0 distance, 0 score → 1.5 distance
            max_score = scores.max() if hasattr(scores, "max") else max(scores)
            results = []
            for (doc, _orig_score), bm25_score in ranked[:top_k]:
                if max_score > 0:
                    pseudo_dist = 1.5 * (1.0 - bm25_score / max_score)
                else:
                    pseudo_dist = 1.5
                results.append((doc, pseudo_dist))

            return results, True
        except Exception:
            return [], False

    # ── Private: Merge + Deduplicate + RRF ───────────────────────────────────

    def _merge(
        self,
        vector_results: List[Tuple[Any, float]],
        bm25_results: List[Tuple[Any, float]],
        top_k: int,
    ) -> Tuple[List[Tuple[Any, float]], int]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion (RRF), then dedup.

        Returns:
            (merged_list, duplicates_removed_count)
        """
        # Build RRF score map keyed by doc id (page_content hash as proxy)
        rrf_scores: Dict[int, Tuple[Any, float, float]] = {}  # id → (doc, vector_score, rrf)

        for rank, (doc, score) in enumerate(vector_results, 1):
            doc_id = id(doc)
            rrf    = 1.0 / (_RRF_K + rank)
            rrf_scores[doc_id] = (doc, score, rrf)

        for rank, (doc, score) in enumerate(bm25_results, 1):
            doc_id = id(doc)
            rrf    = 1.0 / (_RRF_K + rank)
            if doc_id in rrf_scores:
                existing = rrf_scores[doc_id]
                rrf_scores[doc_id] = (existing[0], existing[1], existing[2] + rrf)
            else:
                rrf_scores[doc_id] = (doc, score, rrf)

        # Sort by RRF score descending
        sorted_items = sorted(
            rrf_scores.values(), key=lambda x: x[2], reverse=True
        )[:top_k]

        # Content-based deduplication (identical page_content)
        seen_content: set = set()
        deduped  = []
        dup_count = 0
        for doc, score, _ in sorted_items:
            content_key = doc.page_content[:200]  # use first 200 chars as fingerprint
            if content_key in seen_content:
                dup_count += 1
                continue
            seen_content.add(content_key)
            deduped.append((doc, score))

        return deduped, dup_count

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Simple whitespace + punctuation tokeniser."""
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


# ── Backward-compatibility alias ──────────────────────────────────────────────
# Phase 7.1 code that still imports RetrievalTool / RetrievalResult will work.

class RetrievalResult(KnowledgeRetrievalResult):
    """Backward-compatible alias for KnowledgeRetrievalResult."""
    pass


class RetrievalTool(KnowledgeRetrievalTool):
    """Backward-compatible alias for KnowledgeRetrievalTool."""
    pass
