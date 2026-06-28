"""
Document Analysis Tool  –  Phase 7.1
──────────────────────────────────────
Provides quick structural analysis and metadata extraction from uploaded
medical documents stored in MongoDB / GridFS.

Responsibilities:
  - Retrieve document metadata from MongoDB
  - Download PDF from GridFS to a temp file
  - Count pages and extract raw text statistics
  - Detect document type (lab report, prescription, clinical note, etc.)
  - Identify key findings patterns (CBC values, organ-function markers, etc.)
  - Return a structured summary ready to pass to the LLM for plain-language explanation

The tool does NOT make any LLM calls — it performs purely deterministic
text-pattern matching and statistical analysis.  The LLM is called
*after* by the agent to explain the findings in natural language.

Used by:
  - Document QA Agent  (when analysing a specific uploaded file)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any


# ── Supported document type patterns ─────────────────────────────────────────

_DOC_TYPE_PATTERNS = {
    "CBC Report":             r"\b(complete blood count|cbc|hemoglobin|hematocrit|wbc|rbc|platelet)\b",
    "Blood Chemistry Report": r"\b(creatinine|urea|bun|glucose|sodium|potassium|chloride|bicarbonate|albumin)\b",
    "Liver Function Test":    r"\b(alt|ast|alp|bilirubin|ggt|liver function|lft)\b",
    "Kidney Function Test":   r"\b(creatinine|gfr|bun|urine|kidney|renal|nephrology)\b",
    "Lipid Panel":            r"\b(cholesterol|ldl|hdl|triglycerides|lipid)\b",
    "Thyroid Panel":          r"\b(tsh|t3|t4|thyroid|thyroxine)\b",
    "Diabetes Report":        r"\b(hba1c|fasting glucose|blood sugar|diabetes|insulin)\b",
    "Prescription":           r"\b(rx|prescription|take|twice daily|once daily|tablet|capsule|mg)\b",
    "Radiology Report":       r"\b(x-ray|mri|ct scan|ultrasound|radiology|imaging)\b",
    "Pathology Report":       r"\b(biopsy|pathology|histology|malignant|benign|tumour|tumor)\b",
    "Discharge Summary":      r"\b(discharge|admitted|hospital|inpatient|diagnosis on discharge)\b",
}

# ── Key clinical findings patterns ───────────────────────────────────────────

_FINDING_PATTERNS = {
    # CBC
    "Low Hemoglobin":     r"hemoglobin[\s:]+(\d+\.?\d*)\s*(g/dl|g/l)?",
    "Low Ferritin":       r"ferritin[\s:]+(\d+\.?\d*)\s*(ng/ml|pmol/l)?",
    "Low WBC":            r"wbc[\s:]+(\d+\.?\d*)",
    "High WBC":           r"wbc[\s:]+(\d+\.?\d*)",
    "Low Platelets":      r"platelet[\s:]+(\d+\.?\d*)",
    # Glucose
    "High Blood Sugar":   r"glucose[\s:]+(\d+\.?\d*)\s*(mg/dl|mmol/l)?",
    "High HbA1c":         r"hba1c[\s:]+(\d+\.?\d*)\s*(%)?",
    # Kidney
    "High Creatinine":    r"creatinine[\s:]+(\d+\.?\d*)\s*(mg/dl|umol/l)?",
    "High BUN":           r"\bbun[\s:]+(\d+\.?\d*)",
    # Liver
    "High ALT":           r"\balt[\s:]+(\d+\.?\d*)",
    "High AST":           r"\bast[\s:]+(\d+\.?\d*)",
    # Lipids
    "High Cholesterol":   r"cholesterol[\s:]+(\d+\.?\d*)\s*(mg/dl|mmol/l)?",
    "High LDL":           r"\bldl[\s:]+(\d+\.?\d*)",
    "Low HDL":            r"\bhdl[\s:]+(\d+\.?\d*)",
    # Thyroid
    "Abnormal TSH":       r"\btsh[\s:]+(\d+\.?\d*)",
    # Blood pressure
    "High Blood Pressure":r"blood pressure[\s:]+(\d+/\d+)",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DocumentAnalysisResult:
    """Structured output of the Document Analysis Tool."""

    # Document identity
    doc_id: str
    title: str
    filename: str
    category: str
    organization: str

    # Statistics
    page_count: int
    word_count: int
    char_count: int

    # Classification
    doc_type: str             # e.g. "CBC Report", "Prescription", "Unknown"
    doc_type_confidence: str  # "High" | "Medium" | "Low"

    # Key findings detected by pattern matching
    findings: List[str] = field(default_factory=list)

    # Full extracted text (first 3000 chars for LLM summarisation)
    text_preview: str = ""

    # Error message if extraction failed
    error: str = ""

    @property
    def success(self) -> bool:
        return not bool(self.error)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id":             self.doc_id,
            "title":              self.title,
            "filename":           self.filename,
            "category":           self.category,
            "organization":       self.organization,
            "page_count":         self.page_count,
            "word_count":         self.word_count,
            "char_count":         self.char_count,
            "doc_type":           self.doc_type,
            "doc_type_confidence":self.doc_type_confidence,
            "findings":           self.findings,
        }

    def __str__(self) -> str:
        lines = [
            f"Document   : {self.title}",
            f"File       : {self.filename}",
            f"Category   : {self.category}",
            f"Type       : {self.doc_type} (confidence: {self.doc_type_confidence})",
            f"Pages      : {self.page_count}",
            f"Words      : {self.word_count}",
        ]
        if self.findings:
            lines.append("Key Findings:")
            for f in self.findings:
                lines.append(f"  • {f}")
        else:
            lines.append("Key Findings  : None detected")
        if self.error:
            lines.append(f"Error         : {self.error}")
        return "\n".join(lines)


class DocumentAnalysisTool:
    """
    Extracts structural metadata and clinical finding patterns from uploaded
    medical documents without making any LLM calls.

    Usage:
        tool = DocumentAnalysisTool(doc_service)
        result = tool.run_by_title("blood_report.pdf")
        print(result)
    """

    def __init__(self, doc_service):
        """
        Args:
            doc_service: Initialised DocumentService instance.
        """
        self.doc_service = doc_service

    # ── Public API ────────────────────────────────────────────────────────────

    def run_by_id(self, doc_id: str) -> DocumentAnalysisResult:
        """
        Analyse a document by its MongoDB ObjectId string.

        Args:
            doc_id: MongoDB document _id as string.
        """
        doc_meta = self.doc_service.get_document(doc_id)
        if not doc_meta:
            return self._error_result(doc_id, f"Document with id '{doc_id}' not found.")
        return self._analyse(doc_meta)

    def run_by_title(self, title_fragment: str) -> DocumentAnalysisResult:
        """
        Analyse the first document whose title or filename contains *title_fragment*.

        Args:
            title_fragment: Partial title or filename (case-insensitive).
        """
        all_docs = self.doc_service.list_documents()
        fragment_lower = title_fragment.lower()
        match = None
        for doc in all_docs:
            t = (doc.get("title") or "").lower()
            f = (doc.get("filename") or "").lower()
            if fragment_lower in t or fragment_lower in f:
                match = doc
                break

        if not match:
            return self._error_result(
                title_fragment,
                f"No document found matching '{title_fragment}'."
            )
        return self._analyse(match)

    def list_available(self) -> List[Dict[str, Any]]:
        """
        Return a summary list of all uploaded documents.

        Each entry: {doc_id, title, filename, category, indexed, chunk_count}
        """
        docs = self.doc_service.list_documents()
        return [
            {
                "doc_id":      str(doc.get("_id", "")),
                "title":       doc.get("title", "Unknown"),
                "filename":    doc.get("filename", ""),
                "category":    doc.get("category", "General"),
                "indexed":     doc.get("indexed", False),
                "chunk_count": doc.get("chunk_count", 0),
            }
            for doc in docs
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _analyse(self, doc_meta: dict) -> DocumentAnalysisResult:
        """Download PDF, extract text, classify, and return result."""
        doc_id   = str(doc_meta.get("_id", "unknown"))
        title    = doc_meta.get("title", "Unknown")
        filename = doc_meta.get("filename", "")
        category = doc_meta.get("category", "General")
        org      = doc_meta.get("organization", "General")

        # ── Download PDF to temp ─────────────────────────────────────────────
        try:
            temp_path: Path = self.doc_service.download_document_to_temp(doc_id)
        except Exception as e:
            return self._error_result(doc_id, f"Failed to download document: {e}")

        # ── Extract text with PyPDF ──────────────────────────────────────────
        try:
            text, page_count = self._extract_text(temp_path)
        except Exception as e:
            return self._error_result(doc_id, f"Failed to extract text: {e}")

        # ── Statistics ───────────────────────────────────────────────────────
        word_count = len(text.split())
        char_count = len(text)
        text_lower = text.lower()

        # ── Classify document type ───────────────────────────────────────────
        doc_type, type_confidence = self._classify_doc_type(text_lower)

        # ── Detect key findings ──────────────────────────────────────────────
        findings = self._detect_findings(text_lower)

        # ── Build preview for LLM ────────────────────────────────────────────
        text_preview = text[:3000].strip()

        return DocumentAnalysisResult(
            doc_id=doc_id,
            title=title,
            filename=filename,
            category=category,
            organization=org,
            page_count=page_count,
            word_count=word_count,
            char_count=char_count,
            doc_type=doc_type,
            doc_type_confidence=type_confidence,
            findings=findings,
            text_preview=text_preview,
        )

    @staticmethod
    def _extract_text(pdf_path: Path):
        """Extract text from all pages of a PDF. Returns (full_text, page_count)."""
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader   # fallback for older installs

        reader = PdfReader(str(pdf_path))
        pages = reader.pages
        texts = []
        for page in pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n".join(texts), len(pages)

    @staticmethod
    def _classify_doc_type(text_lower: str):
        """
        Match document type using regex patterns.
        Returns (doc_type_string, confidence_label).
        """
        scores: Dict[str, int] = {}
        for doc_type, pattern in _DOC_TYPE_PATTERNS.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                scores[doc_type] = len(matches)

        if not scores:
            return "General Medical Document", "Low"

        best = max(scores, key=lambda k: scores[k])
        count = scores[best]
        confidence = "High" if count >= 5 else ("Medium" if count >= 2 else "Low")
        return best, confidence

    @staticmethod
    def _detect_findings(text_lower: str) -> List[str]:
        """
        Scan for clinical findings using regex patterns.
        Returns list of human-readable finding strings.
        """
        found = []
        for label, pattern in _FINDING_PATTERNS.items():
            match = re.search(pattern, text_lower)
            if match:
                value = match.group(1) if match.lastindex else None
                entry = label if not value else f"{label}: {value}"
                found.append(entry)
        return found

    @staticmethod
    def _error_result(doc_id: str, error_msg: str) -> DocumentAnalysisResult:
        return DocumentAnalysisResult(
            doc_id=doc_id,
            title="Unknown",
            filename="",
            category="Unknown",
            organization="Unknown",
            page_count=0,
            word_count=0,
            char_count=0,
            doc_type="Unknown",
            doc_type_confidence="Low",
            findings=[],
            text_preview="",
            error=error_msg,
        )
