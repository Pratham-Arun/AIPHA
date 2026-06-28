"""
Phase 7.1 Validation Tests
───────────────────────────
Tests all four tools without requiring a live LLM, database, or ChromaDB.

Run with: .venv\\Scripts\\python.exe test_phase71.py
"""
import sys
sys.path.insert(0, ".")

PASS = "PASS"
FAIL = "FAIL"
failures = []

def check(label, condition):
    status = PASS if condition else FAIL
    if not condition:
        failures.append(label)
    print(f"  [{status}]  {label}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 1 – RetrievalTool
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 1: RetrievalTool")
print("=" * 60)

from tools.retrieval_tool import RetrievalTool, RetrievalResult

class MockRAG:
    def retrieve(self, query, filter_dict=None):
        # Return 2 fake chunks with realistic scores
        from langchain_core.documents import Document
        doc = Document(
            page_content="Hypertension is high blood pressure.",
            metadata={"source": "About High Blood Pressure CDC.pdf",
                      "page": 2, "title": "About High Blood Pressure",
                      "organization": "CDC"}
        )
        results = [(doc, 0.72), (doc, 0.88)]
        context = "Organization: CDC\nDocument: About High Blood Pressure\nPage: 2\n\nHypertension is..."
        return context, results, 1, 42

rt = RetrievalTool(MockRAG())
result = rt.run("What causes hypertension?")

check("RetrievalTool returns RetrievalResult",         isinstance(result, RetrievalResult))
check("has_results is True when chunks present",       result.has_results)
check("final_count >= 1",                              result.final_count >= 1)
check("confidence is valid label",                     result.confidence in ("High", "Medium", "Low"))
check("search_time_ms is populated",                   result.search_time_ms >= 0)
check("context string is non-empty",                   bool(result.context))
check("format_summary on result returns string",       "Knowledge Retrieval" in result.format_summary())

# Empty result
empty = RetrievalTool(MockRAG())
empty.rag = type("E", (), {"retrieve": lambda s,q,filter_dict=None: ("", [], 0, 0)})()
r2 = empty.run("test")
check("has_results is False when no chunks",           not r2.has_results)
check("confidence defaults to Low when empty",         r2.confidence == "Low")


# ═════════════════════════════════════════════════════════════════════════════
# Test 2 – MedicalCalculatorTool
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 2: MedicalCalculatorTool")
print("=" * 60)

from tools.calculator_tool import MedicalCalculatorTool

calc = MedicalCalculatorTool()

# BMI
bmi = calc.bmi(height_cm=181, weight_kg=52)
check("BMI value is correct (15.9)",        round(bmi.bmi, 1) == 15.9)
check("BMI category is Underweight",        "Thinness" in bmi.category or "Underweight" in bmi.category)
check("BMI str output contains 'BMI:'",     "BMI:" in str(bmi))
check("BMI to_dict has bmi key",            "bmi" in bmi.to_dict())

# Normal BMI
bmi_normal = calc.bmi(height_cm=175, weight_kg=70)
check("Normal BMI ~22.9 is Healthy Weight", "Normal" in bmi_normal.category or "Healthy" in bmi_normal.category)

# Overweight
bmi_ow = calc.bmi(height_cm=170, weight_kg=85)
check("Overweight BMI ~29.4 is Overweight", "Overweight" in bmi_ow.category)

# BMR
bmr = calc.bmr(age=30, height_cm=175, weight_kg=70, gender="male")
check("BMR is positive",                    bmr.bmr_calories > 0)
check("BMR has activity_calories dict",     "sedentary" in bmr.activity_calories)
check("Sedentary > BMR",                    bmr.activity_calories["sedentary"] > bmr.bmr_calories)
check("BMR str contains 'Basal Metabolic'", "Basal Metabolic" in str(bmr))

# Female BMR
bmr_f = calc.bmr(age=25, height_cm=165, weight_kg=60, gender="female")
check("Female BMR is lower than male BMR",  bmr_f.bmr_calories < bmr.bmr_calories)

# Water intake
water = calc.water_intake(weight_kg=70)
check("Water intake ~2.45 L for 70 kg",    abs(water.recommended_litres - 2.45) < 0.1)
check("Water to_dict has recommended_ml",  "recommended_ml" in water.to_dict())

# Ideal body weight
ibw = calc.ideal_body_weight(height_cm=175, gender="male")
check("IBW is positive",                    ibw.ideal_weight_kg > 0)
check("IBW str contains 'Ideal'",           "Ideal" in str(ibw))

# BSA
bsa = calc.body_surface_area(height_cm=175, weight_kg=70)
check("BSA ~1.85 m² for 175cm/70kg",       abs(bsa.bsa_m2 - 1.850) < 0.05)

# Dispatcher
d_result = calc.run("bmi", height_cm=170, weight_kg=65)
check("Dispatcher bmi works",               hasattr(d_result, "bmi"))

# Validation errors
try:
    calc.bmi(height_cm=-10, weight_kg=70)
    check("Negative height raises ValueError", False)
except ValueError:
    check("Negative height raises ValueError", True)

try:
    calc.bmi(height_cm=170, weight_kg=0)
    check("Zero weight raises ValueError",     False)
except ValueError:
    check("Zero weight raises ValueError",     True)

try:
    calc.run("unknown_calc", height_cm=170)
    check("Unknown calc raises ValueError",    False)
except ValueError:
    check("Unknown calc raises ValueError",    True)


# ═════════════════════════════════════════════════════════════════════════════
# Test 3 – DocumentAnalysisTool (mock doc_service)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 3: DocumentAnalysisTool")
print("=" * 60)

from tools.document_analysis_tool import DocumentAnalysisTool, DocumentAnalysisResult
import tempfile, os

class MockDocService:
    def get_document(self, doc_id):
        return {
            "_id": doc_id, "title": "CBC Blood Report", "filename": "cbc.pdf",
            "category": "Lab Report", "organization": "General",
            "gridfs_id": "fake_gridfs_id"
        }
    def list_documents(self):
        return [{"_id": "doc1", "title": "CBC Blood Report", "filename": "cbc.pdf",
                 "category": "Lab Report", "indexed": True, "chunk_count": 10}]
    def download_document_to_temp(self, doc_id):
        # Write a fake text-based PDF using reportlab if available, else write a
        # minimal valid PDF that pypdf can parse
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        # Minimal PDF with CBC-related text
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
            b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length 120 >>\nstream\n"
            b"BT /F1 12 Tf 50 750 Td "
            b"(CBC Report Hemoglobin 10.5 g/dL WBC 4.2 Platelet 150 Ferritin 8 ng/ml) Tj ET\n"
            b"endstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n0000000000 65535 f\n"
            b"0000000009 00000 n\n0000000058 00000 n\n"
            b"0000000115 00000 n\n0000000274 00000 n\n0000000448 00000 n\n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n528\n%%EOF"
        )
        tmp.write(pdf_content)
        tmp.flush()
        tmp.close()
        return type("P", (), {"__str__": lambda s: tmp.name, "__fspath__": lambda s: tmp.name})()

dat = DocumentAnalysisTool(MockDocService())

# list_available
available = dat.list_available()
check("list_available returns list",         isinstance(available, list))
check("list_available entry has title",      available and "title" in available[0])

# error result for unknown doc
err = dat.run_by_title("nonexistent_document_xyz")
check("run_by_title returns error result",   not err.success)
check("error result has error message",      bool(err.error))

# to_dict and __str__ on error result
d = err.to_dict()
check("error to_dict has doc_type key",     "doc_type" in d)
check("error __str__ has 'Document'",       "Document" in str(err))


# ═════════════════════════════════════════════════════════════════════════════
# Test 4 – SystemStatusTool
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 4: SystemStatusTool")
print("=" * 60)

from tools.system_status_tool import SystemStatusTool, SystemStatusResult, ComponentStatus

class MockWorkflow:
    graph = object()  # non-None

class MockRagStatus:
    def get_document_count(self): return 6061

class MockDocServiceStatus:
    class gridfs_mgr: pass
    def list_documents(self): return [{}] * 51
    @property
    def gridfs_mgr(self): return self

st = SystemStatusTool(
    rag=MockRagStatus(),
    doc_service=MockDocServiceStatus(),
    workflow=MockWorkflow(),
    llm=None,
)
result = st.run()

check("run() returns SystemStatusResult",        isinstance(result, SystemStatusResult))
check("MongoDB component present",               "MongoDB" in result.components)
check("GridFS component present",                "GridFS" in result.components)
check("ChromaDB component present",              "ChromaDB" in result.components)
check("LangGraph component present",             "LangGraph" in result.components)
check("Gemini component present",                "Gemini" in result.components)
check("document_count populated (51)",           result.document_count == 51)
check("chunk_count populated (6061)",            result.chunk_count == 6061)
check("agent_count is 6",                        result.agent_count == 6)
check("check_time_ms is non-negative",           result.check_time_ms >= 0)
check("__str__ contains 'System Diagnostics'",   "System Diagnostics" in str(result))
check("__str__ contains 'Documents'",            "Documents" in str(result))

# ComponentStatus
cs = ComponentStatus("Test", "Healthy", latency_ms=5)
check("ComponentStatus.is_healthy True",         cs.is_healthy)
cs2 = ComponentStatus("Test2", "Unavailable")
check("ComponentStatus.is_healthy False",        not cs2.is_healthy)


# ═════════════════════════════════════════════════════════════════════════════
# Test 5 – Graph compiles with tool_node
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 5: Graph Compilation with tool_node")
print("=" * 60)

from langchain_core.runnables import RunnableLambda
from graph.builder import build_graph
from memory import MemoryManager

mock_llm = RunnableLambda(lambda x: "mock")
memory   = MemoryManager()
graph    = build_graph(mock_llm, memory, MockRagStatus())

nodes = list(graph.get_graph().nodes.keys())
check("tool_node registered in graph",           "tool_node" in nodes)
check("agent_execution_node registered",         "agent_execution_node" in nodes)
check("formatter_node registered",               "formatter_node" in nodes)
check("supervisor_node registered",              "supervisor_node" in nodes)
print(f"  [INFO] Nodes: {nodes}")


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
if failures:
    print(f"RESULT: {len(failures)} test(s) FAILED")
    for f in failures:
        print(f"  FAIL: {f}")
    sys.exit(1)
else:
    print(f"RESULT: ALL TESTS PASSED ✓")
print("=" * 60)
