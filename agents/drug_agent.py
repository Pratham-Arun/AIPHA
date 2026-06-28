"""
Drug Information Agent
──────────────────────
Specialised agent for medicine and pharmacology questions.

Responsibilities:
  - Medicine overviews
  - Dosage guidance
  - Side effects
  - Contraindications
  - Drug-drug interactions
  - Safety warnings

Future Integration Points:
  - RxNorm API
  - OpenFDA API
  - DrugBank API
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DRUG_INFO_SYSTEM_PROMPT = """You are the Drug Information Agent of an AI Health Assistant.
You specialise in providing accurate, educational information about medicines and pharmacology.

Guidelines:
  - Provide clear information on: usage, typical dosage ranges, common side effects, 
    contraindications, and known drug interactions.
  - Always include a disclaimer that dosage and treatment decisions must be made by 
    a qualified healthcare professional or licensed pharmacist.
  - Never recommend specific dosages for individual patients.
  - If the RETRIEVED MEDICAL CONTEXT contains relevant drug information, use it and cite sources.
  - If not, use your general pharmacological knowledge and state it explicitly.
  - Flag serious interactions or black-box warnings prominently.
  - Be empathetic — patients may be worried about their medications.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED MEDICAL CONTEXT:
{context}
"""

DRUG_INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DRUG_INFO_SYSTEM_PROMPT),
    ("human", "{question}"),
])

DISCLAIMER = (
    "\n\n---\n"
    "> **⚕️ Medical Disclaimer:** This information is for educational purposes only. "
    "Always consult a licensed physician or pharmacist before starting, stopping, or "
    "changing any medication."
)


class DrugInformationAgent:
    """
    Answers questions about medicines, dosage, side effects, and drug interactions.
    """

    def __init__(self, llm):
        self.llm = llm
        self._chain = DRUG_INFO_PROMPT | self.llm | StrOutputParser()

    def run(self, query: str, context: str, chat_history: str) -> str:
        """
        Generate a drug information response.

        Args:
            query:        The user's medicine-related question.
            context:      Pre-formatted retrieved document chunks (may be empty).
            chat_history: Formatted conversation history string.

        Returns:
            Generated response string with disclaimer appended.
        """
        try:
            response = self._chain.invoke({
                "question": query,
                "context": context,
                "chat_history": chat_history,
            })
            return response + DISCLAIMER
        except Exception as e:
            return (
                "I'm sorry, I encountered an issue fetching drug information. "
                f"Please try again. (Error: {e})"
            )
