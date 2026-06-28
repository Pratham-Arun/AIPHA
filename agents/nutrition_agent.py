"""
Nutrition Agent  –  Phase 7 / 7.1
───────────────────────────────────
Specialised agent for dietary and nutritional guidance.

Phase 7.1 enhancement:
  The agent now accepts an optional *calculator_output* string produced by
  MedicalCalculatorTool (BMI, BMR, water intake, etc.).  When present, the
  LLM is asked to *explain* the deterministic result rather than compute it,
  which eliminates calculation hallucinations.

Responsibilities:
  - Meal planning and diet recommendations
  - Disease-specific diets (diabetes, kidney disease, hypertension, etc.)
  - Vitamin and mineral deficiency information
  - Weight management guidance
  - Healthy eating habits
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

NUTRITION_SYSTEM_PROMPT = """You are the Nutrition Agent of an AI Health Assistant.
You specialise in providing evidence-based nutritional guidance and dietary recommendations.

Guidelines:
  - Offer practical, actionable meal and diet advice.
  - When a disease-specific diet is requested (e.g., diabetic diet, renal diet),
    explain key principles clearly.
  - Always recommend consulting a Registered Dietitian (RD) for personalised meal plans.
  - If the RETRIEVED MEDICAL CONTEXT contains relevant nutritional information, use it
    and cite sources.
  - If not, answer from your general nutritional knowledge and state it explicitly.
  - Be positive and encouraging — healthy eating should feel achievable.
  - Include practical tips: food swaps, portion guidance, foods to prefer/limit.
  - If CALCULATOR RESULTS are provided, use them as the authoritative numbers and
    explain what they mean in plain language. Do NOT recalculate.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED MEDICAL CONTEXT:
{context}

CALCULATOR RESULTS (deterministic — use these exact numbers):
{calculator_output}
"""

NUTRITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", NUTRITION_SYSTEM_PROMPT),
    ("human", "{question}"),
])

DISCLAIMER = (
    "\n\n---\n"
    "> **🥗 Nutrition Disclaimer:** These recommendations are for general educational "
    "purposes. For a personalised nutrition plan tailored to your health conditions, "
    "please consult a Registered Dietitian or healthcare provider."
)


class NutritionAgent:
    """
    Answers questions about diet, nutrition, meal planning, and food choices.
    In Phase 7.1, accepts pre-computed calculator results to eliminate
    LLM arithmetic hallucinations.
    """

    def __init__(self, llm):
        self.llm = llm
        self._chain = NUTRITION_PROMPT | self.llm | StrOutputParser()

    def run(
        self,
        query: str,
        context: str,
        chat_history: str,
        calculator_output: str = "",
    ) -> str:
        """
        Generate a nutrition-focused response.

        Args:
            query:             The user's nutrition / diet question.
            context:           Pre-formatted retrieved document chunks (may be empty).
            chat_history:      Formatted conversation history string.
            calculator_output: Pre-computed result string from MedicalCalculatorTool
                               (empty string if no calculation was performed).

        Returns:
            Generated response string with disclaimer appended.
        """
        calc_section = calculator_output if calculator_output else "No calculations performed."
        try:
            response = self._chain.invoke({
                "question":          query,
                "context":           context,
                "chat_history":      chat_history,
                "calculator_output": calc_section,
            })
            return response + DISCLAIMER
        except Exception as e:
            return (
                "I'm sorry, I encountered an issue generating nutrition advice. "
                f"Please try again. (Error: {e})"
            )
