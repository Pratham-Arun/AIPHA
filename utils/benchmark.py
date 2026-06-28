"""
Hybrid Search Benchmark  –  Phase 7.3
───────────────────────────────────────
Compares Vector-only search vs Hybrid (Vector + BM25) search on a
fixed set of 40 medical queries drawn from the actual document corpus.

For each query the benchmark checks whether the top retrieved chunk
comes from the expected document (keyword match in title/source).

Results are printed as a Markdown table and saved to
  benchmark_results.json

Run standalone:
    .venv\\Scripts\\python.exe utils/benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Benchmark query set  (40 queries, 5 categories) ──────────────────────────
#
# Each entry: (query, expected_keyword_in_source, category)
#
# expected_keyword: a lowercase substring that should appear in at least one
# retrieved document's source path / title.  Empty string means "any result".
#
BENCHMARK_QUERIES: List[Tuple[str, str, str]] = [
    # ── General Medical / Disease ────────────────────────────────────────────
    ("What causes hypertension?",                    "blood pressure",    "Medical"),
    ("What is coronary artery disease?",             "coronary",          "Medical"),
    ("What are the symptoms of heart attack?",       "heart attack",      "Medical"),
    ("How does stroke affect the brain?",            "stroke",            "Medical"),
    ("What is cardiac arrest?",                      "cardiac",           "Medical"),
    ("Explain chronic kidney disease",               "kidney",            "Medical"),
    ("What is the relationship between diabetes and heart disease?", "diabetes", "Medical"),
    ("What causes high blood pressure during pregnancy?", "blood pressure", "Medical"),
    ("How does smoking affect cardiovascular health?", "cardiovascular",  "Medical"),
    ("What is behavioral health?",                   "behavioral",        "Medical"),

    # ── Abbreviation / Keyword-heavy (BM25 advantage) ────────────────────────
    ("HbA1c level in diabetes management",           "diabetes",          "Abbreviation"),
    ("LDL and HDL cholesterol levels",               "cardiovascular",    "Abbreviation"),
    ("BMI obesity classification",                   "nutrition",         "Abbreviation"),
    ("CDC hypertension statistics",                  "blood pressure",    "Abbreviation"),
    ("WHO guidelines for diabetes",                  "diabetes",          "Abbreviation"),
    ("GFR kidney function test",                     "kidney",            "Abbreviation"),
    ("CBC complete blood count report",              "diabetes",          "Abbreviation"),
    ("TSH thyroid stimulating hormone",              "diabetes",          "Abbreviation"),
    ("CKD chronic kidney disease stages",            "kidney",            "Abbreviation"),
    ("CAD risk factors in men",                      "coronary",          "Abbreviation"),

    # ── Nutrition / Diet ─────────────────────────────────────────────────────
    ("Foods rich in iron",                           "nutrition",         "Nutrition"),
    ("Diet plan for diabetes",                       "diabetes",          "Nutrition"),
    ("Benefits of healthy eating for adults",        "nutrition",         "Nutrition"),
    ("What micronutrients does the body need?",      "micronutrients",    "Nutrition"),
    ("Healthy food environments",                    "nutrition",         "Nutrition"),
    ("Foods to avoid for infants",                   "infant",            "Nutrition"),
    ("Breastfeeding and nutrition",                  "breastfeeding",     "Nutrition"),
    ("What are the benefits of healthy eating for children?", "nutrition", "Nutrition"),
    ("Vitamins and minerals overview",               "micronutrients",    "Nutrition"),
    ("Diet for kidney disease patients",             "kidney",            "Nutrition"),

    # ── Mental Health ────────────────────────────────────────────────────────
    ("What is mental health?",                       "mental health",     "Mental Health"),
    ("Anxiety and depression in children",           "mental health",     "Mental Health"),
    ("Behavioral problems in children",              "mental health",     "Mental Health"),
    ("How to manage stress?",                        "mental health",     "Mental Health"),
    ("Children's mental health overview",            "mental health",     "Mental Health"),

    # ── Organization-specific (WHO/CDC boosting) ─────────────────────────────
    ("WHO report on diabetes",                       "diabetes",          "Org-Specific"),
    ("CDC statistics on heart disease",              "heart disease",     "Org-Specific"),
    ("CDC information on stroke prevention",         "stroke",            "Org-Specific"),
    ("CDC chronic kidney disease facts",             "kidney",            "Org-Specific"),
    ("CDC cardiovascular disease indicators",        "cardiovascular",    "Org-Specific"),
]


@dataclass
class QueryResult:
    query: str
    category: str
    expected_keyword: str
    vector_hit: bool = False
    hybrid_hit: bool = False
    vector_top_source: str = ""
    hybrid_top_source: str = ""
    vector_time_ms: int = 0
    hybrid_time_ms: int = 0


@dataclass
class BenchmarkReport:
    results: List[QueryResult] = field(default_factory=list)
    vector_accuracy: float = 0.0
    hybrid_accuracy: float = 0.0
    vector_avg_ms: float = 0.0
    hybrid_avg_ms: float = 0.0
    total_queries: int = 0

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 72,
            "  Hybrid Search Benchmark Results  –  Phase 7.3",
            "=" * 72,
            f"  Total Queries   : {self.total_queries}",
            f"  Vector Accuracy : {self.vector_accuracy:.1f}%",
            f"  Hybrid Accuracy : {self.hybrid_accuracy:.1f}%",
            f"  Improvement     : +{self.hybrid_accuracy - self.vector_accuracy:.1f}%",
            f"  Vector Avg Time : {self.vector_avg_ms:.0f} ms",
            f"  Hybrid Avg Time : {self.hybrid_avg_ms:.0f} ms",
            "=" * 72,
            "",
            f"  {'#':<3} {'Category':<14} {'Query':<42} {'Vector':<8} {'Hybrid':<8}",
            "  " + "-" * 70,
        ]
        for i, r in enumerate(self.results, 1):
            v = "✅" if r.vector_hit else "❌"
            h = "✅" if r.hybrid_hit else "❌"
            q = r.query[:40] + ".." if len(r.query) > 42 else r.query
            lines.append(f"  {i:<3} {r.category:<14} {q:<42} {v:<8} {h:<8}")
        lines += [
            "  " + "-" * 70,
            f"  {'TOTAL':<17} {'':<42} "
            f"{sum(r.vector_hit for r in self.results)}/{self.total_queries:<5} "
            f"{sum(r.hybrid_hit for r in self.results)}/{self.total_queries}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_queries":    self.total_queries,
            "vector_accuracy":  round(self.vector_accuracy, 2),
            "hybrid_accuracy":  round(self.hybrid_accuracy, 2),
            "improvement":      round(self.hybrid_accuracy - self.vector_accuracy, 2),
            "vector_avg_ms":    round(self.vector_avg_ms, 1),
            "hybrid_avg_ms":    round(self.hybrid_avg_ms, 1),
            "results": [
                {
                    "query":              r.query,
                    "category":           r.category,
                    "expected_keyword":   r.expected_keyword,
                    "vector_hit":         r.vector_hit,
                    "hybrid_hit":         r.hybrid_hit,
                    "vector_top_source":  r.vector_top_source,
                    "hybrid_top_source":  r.hybrid_top_source,
                    "vector_time_ms":     r.vector_time_ms,
                    "hybrid_time_ms":     r.hybrid_time_ms,
                }
                for r in self.results
            ],
        }


class HybridSearchBenchmark:
    """
    Runs the 40-query benchmark against a live RAG pipeline.

    Usage:
        from utils.benchmark import HybridSearchBenchmark
        bench = HybridSearchBenchmark(rag_pipeline)
        report = bench.run()
        print(report)
        bench.save(report)
    """

    OUTPUT_FILE = "benchmark_results.json"

    def __init__(self, rag_pipeline, top_k: int = 6):
        self.rag   = rag_pipeline
        self.top_k = top_k

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        queries: Optional[List[Tuple[str, str, str]]] = None,
        verbose: bool = True,
    ) -> BenchmarkReport:
        """
        Execute the full benchmark.

        Args:
            queries: Override the default BENCHMARK_QUERIES list.
            verbose: Print progress to stdout.

        Returns:
            BenchmarkReport with per-query and aggregate results.
        """
        from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool

        qs     = queries or BENCHMARK_QUERIES
        krt    = KnowledgeRetrievalTool(self.rag)
        report = BenchmarkReport(total_queries=len(qs))

        if verbose:
            print(f"\n  Running benchmark on {len(qs)} queries …\n")

        for i, (query, expected_kw, category) in enumerate(qs, 1):
            if verbose:
                print(f"  [{i:>2}/{len(qs)}] {query[:55]}", end="  ", flush=True)

            # ── Vector-only search ────────────────────────────────────────────
            t0 = time.time()
            try:
                _, vec_results, _, _ = self.rag.retrieve(query)
            except Exception:
                vec_results = []
            vec_ms = int((time.time() - t0) * 1000)

            vec_top_src = self._top_source(vec_results)
            vec_hit     = self._check_hit(vec_results, expected_kw)

            # ── Hybrid search ─────────────────────────────────────────────────
            t0 = time.time()
            try:
                krt.invalidate_bm25_cache()
                hyb = krt.run(query)
                hyb_results = hyb.chunks
            except Exception:
                hyb_results = []
            hyb_ms = int((time.time() - t0) * 1000)

            hyb_top_src = self._top_source(hyb_results)
            hyb_hit     = self._check_hit(hyb_results, expected_kw)

            qr = QueryResult(
                query              = query,
                category           = category,
                expected_keyword   = expected_kw,
                vector_hit         = vec_hit,
                hybrid_hit         = hyb_hit,
                vector_top_source  = vec_top_src,
                hybrid_top_source  = hyb_top_src,
                vector_time_ms     = vec_ms,
                hybrid_time_ms     = hyb_ms,
            )
            report.results.append(qr)

            if verbose:
                v = "✅" if vec_hit else "❌"
                h = "✅" if hyb_hit else "❌"
                print(f"Vector {v}  Hybrid {h}", flush=True)

        # ── Aggregate ─────────────────────────────────────────────────────────
        n = len(report.results)
        if n > 0:
            report.vector_accuracy = (
                sum(r.vector_hit for r in report.results) / n * 100
            )
            report.hybrid_accuracy = (
                sum(r.hybrid_hit for r in report.results) / n * 100
            )
            report.vector_avg_ms = (
                sum(r.vector_time_ms for r in report.results) / n
            )
            report.hybrid_avg_ms = (
                sum(r.hybrid_time_ms for r in report.results) / n
            )

        return report

    def save(self, report: BenchmarkReport) -> None:
        """Save benchmark results to benchmark_results.json."""
        try:
            with open(self.OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\n  Saved to {self.OUTPUT_FILE}")
        except OSError as e:
            print(f"\n  Could not save benchmark: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _top_source(results: List[Tuple[Any, float]]) -> str:
        if not results:
            return ""
        doc, _ = results[0]
        src = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "")
        return (title or src).lower()

    @staticmethod
    def _check_hit(
        results: List[Tuple[Any, float]],
        keyword: str,
    ) -> bool:
        """
        Return True if *keyword* appears in the source/title of any
        retrieved chunk (case-insensitive).  Empty keyword → always True.
        """
        if not keyword:
            return bool(results)
        kw = keyword.lower()
        for doc, _ in results:
            src   = (doc.metadata.get("source",       "") or "").lower()
            title = (doc.metadata.get("title",         "") or "").lower()
            org   = (doc.metadata.get("organization",  "") or "").lower()
            text  = doc.page_content.lower()[:400]
            if kw in src or kw in title or kw in text or kw in org:
                return True
        return False


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _run_cli():
    """Run the benchmark directly against the live database."""
    print("=" * 62)
    print("  Hybrid Search Benchmark  –  Phase 7.3")
    print("=" * 62)

    import config
    from database import MongoConnection
    from rag import RAGPipeline

    try:
        config.validate_config()
        MongoConnection.validate_connection()
    except Exception as e:
        print(f"\n  Error: {e}")
        print("  Ensure MongoDB is running and GOOGLE_API_KEY is set.\n")
        sys.exit(1)

    print("\n  Initialising RAG pipeline …")
    rag = RAGPipeline()
    rag.initialize()

    count = rag.get_document_count()
    if count == 0:
        print("\n  WARNING: ChromaDB is empty. Upload documents first.\n")
        print("  Benchmark cannot run without indexed documents.\n")
        sys.exit(1)

    print(f"  ChromaDB: {count} chunks loaded.")

    bench  = HybridSearchBenchmark(rag)
    report = bench.run(verbose=True)

    print(report)
    bench.save(report)
    rag.shutdown()


if __name__ == "__main__":
    _run_cli()
