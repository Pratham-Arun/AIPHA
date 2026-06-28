"""
Supervisor Agent
────────────────
Central controller of the multi-agent system.

Responsibilities:
  - Analyse the user query and detected intent
  - Select the most appropriate specialized agent
  - Decide whether document retrieval is required
  - Route fallback / emergency queries appropriately

Agent types returned:
  "document_qa"    – medical questions grounded in uploaded documents
  "drug_info"      – medicine, dosage, side effects, interactions
  "nutrition"      – diet, meal plans, vitamins, weight management
  "mental_health"  – stress, anxiety, sleep, mindfulness (educational)
  "general_health" – greetings, general wellness, exercise, small-talk
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

SUPERVISOR_PROMPT = ChatPromptTemplate.from_template(
    """You are the Supervisor Agent of an AI Health Assistant system.
Your ONLY job is to select the correct specialized agent for the user's query.

Available agents:
  - document_qa    : Medical questions about diseases, conditions, symptoms, treatments, 
                     health statistics, cardiac health, kidney disease, stroke, blood pressure,
                     breastfeeding, mental health overview, diabetes overview
  - drug_info      : Questions specifically about medicines, drugs, dosage, side effects, 
                     drug interactions, contraindications, safety warnings, pharmacy questions
  - nutrition      : Diet plans, meal planning, food recommendations, vitamins, minerals, 
                     nutritional deficiencies, weight management, disease-specific diets, 
                     healthy eating, infant/toddler nutrition, food to avoid
  - mental_health  : Stress management, anxiety, depression (educational), sleep hygiene, 
                     mindfulness, mental wellness, emotional wellbeing, relaxation techniques
  - general_health : Greetings, general wellness, exercise, preventive healthcare, 
                     lifestyle tips, fitness, small talk, off-topic queries

Detected intent: {intent}
User query: {query}

Rules:
  1. Respond with ONLY the agent name (one of: document_qa, drug_info, nutrition, mental_health, general_health)
  2. Do NOT add any explanation, punctuation, or extra words
  3. If the query mentions a specific drug/medicine/medication by name → drug_info
  4. If the query asks for a diet, meal plan, or food recommendation → nutrition
  5. If the query mentions stress, anxiety, depression, sleep, or mindfulness → mental_health
  6. If the query is a greeting or unrelated to health → general_health
  7. For all other medical questions → document_qa

Selected agent:"""
)


class SupervisorAgent:
    """
    Routes user queries to the appropriate specialized agent.
    """

    def __init__(self, llm):
        self.llm = llm
        self._chain = SUPERVISOR_PROMPT | self.llm | StrOutputParser()

        # Known intents that map directly without LLM routing
        self._intent_map = {
            "Greeting": "general_health",
            "General Chat": "general_health",
            "Emergency": "general_health",
            "Follow-up Question": "document_qa",  # default; LLM may override
        }

        # Keywords for fast-path routing (avoids an LLM call)
        self._drug_keywords = {
            "metformin", "aspirin", "ibuprofen", "paracetamol", "acetaminophen",
            "amoxicillin", "lisinopril", "atorvastatin", "omeprazole", "mg",
            "dosage", "dose", "tablet", "capsule", "prescription", "drug",
            "medicine", "medication", "side effect", "interaction", "overdose",
            "antibiotic", "insulin", "vaccine", "injection", "pill",
        }
        self._nutrition_keywords = {
            "diet", "meal", "food", "eat", "nutrition", "vitamin", "mineral",
            "calories", "protein", "carb", "fat", "weight", "obese", "vegetarian",
            "vegan", "iron", "calcium", "supplement", "nutrient", "recipe",
        }
        self._mental_keywords = {
            "stress", "anxiety", "anxious", "depressed", "depression", "sleep",
            "insomnia", "mental health", "mindfulness", "meditate", "meditation",
            "panic", "worry", "mood", "emotional", "therapy", "relax", "burnout",
            "sad", "happiness", "wellbeing",
        }

    def select_agent(self, query: str, intent: str) -> str:
        """
        Determines which specialized agent should handle the query.

        Returns one of:
          "document_qa" | "drug_info" | "nutrition" | "mental_health" | "general_health"
        """
        # 1. Fast-path intent overrides
        if intent in ("Greeting", "General Chat", "Emergency"):
            return self._intent_map[intent]

        # 2. Explicit intent mappings
        intent_direct = {
            "Medicine Information": "drug_info",
            "Nutrition": "nutrition",
            "Mental Health": "mental_health",
            "Medical Question": "document_qa",
            "Medical Report Analysis": "document_qa",
        }
        if intent in intent_direct:
            return intent_direct[intent]

        # 3. Keyword fast-path on query text
        query_lower = query.lower()
        if any(kw in query_lower for kw in self._drug_keywords):
            return "drug_info"
        if any(kw in query_lower for kw in self._nutrition_keywords):
            return "nutrition"
        if any(kw in query_lower for kw in self._mental_keywords):
            return "mental_health"

        # 4. LLM-based routing for ambiguous queries
        try:
            result = self._chain.invoke({"query": query, "intent": intent}).strip()
            result = result.lower().replace(".", "").replace('"', "").strip()
            valid = {"document_qa", "drug_info", "nutrition", "mental_health", "general_health"}
            return result if result in valid else "document_qa"
        except Exception:
            return "document_qa"

    def requires_retrieval(self, agent_type: str) -> bool:
        """Returns True if this agent type should trigger document retrieval."""
        return agent_type in {"document_qa", "drug_info", "nutrition"}
