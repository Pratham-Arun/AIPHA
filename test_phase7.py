"""
Phase 7 Validation Tests
Tests the Supervisor routing logic and FormatterAgent without
requiring a live LLM, database, or ChromaDB connection.

Run with: .venv\\Scripts\\python.exe test_phase7.py
"""
import sys
sys.path.insert(0, ".")

from langchain_core.runnables import RunnableLambda

# ── Mock LLM compatible with LangChain pipe operator ─────────────────────────
def _mock_invoke(inputs):
    return "document_qa"

MockLLM = RunnableLambda(_mock_invoke)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: SupervisorAgent routing
# ─────────────────────────────────────────────────────────────────────────────
from agents.supervisor_agent import SupervisorAgent

sup = SupervisorAgent(MockLLM)

routing_tests = [
    # (query,                                       intent,               expected_agent)
    ("Hello",                                       "Greeting",            "general_health"),
    ("What causes hypertension?",                   "Medical Question",    "document_qa"),
    ("Foods rich in iron",                          "Nutrition",           "nutrition"),
    ("Can I take Aspirin with Ibuprofen?",          "Medicine Information","drug_info"),
    ("I am feeling stressed",                       "Mental Health",       "mental_health"),
    ("Create a diet plan for diabetes",             "Nutrition",           "nutrition"),
    ("What is Metformin?",                          "Medicine Information","drug_info"),
    ("How to improve sleep?",                       "Mental Health",       "mental_health"),
    ("Hi there!",                                   "Greeting",            "general_health"),
    ("How much water should I drink per day?",      "General Chat",        "general_health"),
]

print("=" * 60)
print("Test 1: SupervisorAgent – Routing Logic")
print("=" * 60)
failures = []
for query, intent, expected in routing_tests:
    result = sup.select_agent(query, intent)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        failures.append((query, expected, result))
    print(f"  [{status}]  \"{query[:45]}\"")
    if status == "FAIL":
        print(f"         Expected: {expected}  Got: {result}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: requires_retrieval
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 2: SupervisorAgent – requires_retrieval")
print("=" * 60)
retrieval_tests = [
    ("document_qa",    True),
    ("drug_info",      True),
    ("nutrition",      True),
    ("mental_health",  False),
    ("general_health", False),
]
for agent_type, expected in retrieval_tests:
    result = sup.requires_retrieval(agent_type)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        failures.append((f"requires_retrieval({agent_type})", expected, result))
    print(f"  [{status}]  {agent_type} => {result}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: MentalHealthAgent crisis detection
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 3: MentalHealthAgent – Crisis Detection")
print("=" * 60)
from agents.mental_health_agent import MentalHealthAgent

mh = MentalHealthAgent(MockLLM)
crisis_response = mh.run("I want to kill myself", context="", chat_history="")
is_crisis = "988" in crisis_response or "crisis" in crisis_response.lower()
status = "PASS" if is_crisis else "FAIL"
print(f"  [{status}]  Crisis query triggers emergency resources: {is_crisis}")
if not is_crisis:
    failures.append(("crisis detection", True, False))

safe_response_ok = "988" not in mh.run("How do I reduce stress?", context="", chat_history="")
# Note: safe queries go through the real LLM chain which is mocked,
# so we just check the crisis path is correctly bypassed.
print(f"  [INFO]  Non-crisis query does not hard-code 988 in prefix: {safe_response_ok}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: FormatterAgent output structure
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 4: FormatterAgent – Output Structure")
print("=" * 60)
from agents.formatter_agent import FormatterAgent

fmt = FormatterAgent()
out = fmt.format(
    response="Hypertension is high blood pressure.",
    agent_type="document_qa",
    sources=[],
    processing_time=1.23,
    intent="Medical Question",
)
has_label  = "Document QA Agent" in out
has_timing = "1.23" in out
print(f"  [{'PASS' if has_label else 'FAIL'}]  Agent label present in output")
print(f"  [{'PASS' if has_timing else 'FAIL'}]  Processing time present in output")
if not has_label:
    failures.append(("formatter label", True, False))
if not has_timing:
    failures.append(("formatter timing", True, False))

# With mock sources
class MockDoc:
    def __init__(self):
        self.metadata = {
            "source": "Hypertension CDC.pdf",
            "page": 3,
            "title": "About High Blood Pressure",
            "organization": "CDC",
        }
        self.page_content = "Hypertension affects 1 in 3 adults."

sources = [(MockDoc(), 0.75), (MockDoc(), 0.95)]
out_with_sources = fmt.format(
    response="High blood pressure...",
    agent_type="document_qa",
    sources=sources,
    processing_time=2.0,
)
has_citations = "Sources" in out_with_sources
print(f"  [{'PASS' if has_citations else 'FAIL'}]  Citations block present when sources provided")
if not has_citations:
    failures.append(("formatter citations", True, False))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if failures:
    print(f"RESULT: {len(failures)} test(s) FAILED")
    for name, expected, got in failures:
        print(f"  FAIL: {name} — expected {expected}, got {got}")
    sys.exit(1)
else:
    print("RESULT: ALL TESTS PASSED ✓")
print("=" * 60)
