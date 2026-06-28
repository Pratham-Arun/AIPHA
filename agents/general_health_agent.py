"""
General Health Agent
────────────────────
Handles general wellness queries, greetings, and off-topic messages.

Responsibilities:
  - Greetings and conversational small-talk
  - General wellness and lifestyle tips
  - Exercise and physical activity guidance
  - Preventive healthcare reminders
  - Hydration, sleep basics, and hygiene tips
  - Fallback for queries that don't fit other specialised agents
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GENERAL_HEALTH_SYSTEM_PROMPT = """You are the General Health Agent of an AI Health Assistant.
You handle general wellness questions, greetings, lifestyle tips, and preventive health guidance.

Guidelines:
  - Be friendly, warm, and approachable.
  - For greetings, respond naturally and invite the user to ask health questions.
  - Provide evidence-based general wellness advice: hydration, sleep, exercise, hygiene.
  - For preventive health, share practical reminders: regular check-ups, screenings, vaccinations.
  - Keep responses concise and actionable.
  - If a question is outside your scope (complex medical query), gently suggest the user 
    ask more specifically so you can route them to the right specialist.
  - Never diagnose conditions or recommend specific treatments.

CONVERSATION HISTORY:
{chat_history}
"""

GENERAL_HEALTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GENERAL_HEALTH_SYSTEM_PROMPT),
    ("human", "{question}"),
])


class GeneralHealthAgent:
    """
    Handles greetings, general wellness, and preventive health queries.
    """

    def __init__(self, llm):
        self.llm = llm
        self._chain = GENERAL_HEALTH_PROMPT | self.llm | StrOutputParser()

    def run(self, query: str, chat_history: str) -> str:
        """
        Generate a general health / greeting response.

        Args:
            query:        The user's message.
            chat_history: Formatted conversation history string.

        Returns:
            Generated response string.
        """
        try:
            return self._chain.invoke({
                "question": query,
                "chat_history": chat_history,
            })
        except Exception as e:
            return (
                "Hello! I'm your AI Health Assistant. I'm here to help with health "
                "questions, wellness tips, and general medical information. "
                f"How can I help you today? (Error recovering: {e})"
            )
