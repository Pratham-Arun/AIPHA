"""
Mental Health Agent
───────────────────
Educational mental wellness assistant.

Responsibilities:
  - Stress management techniques
  - Anxiety education (non-diagnostic)
  - Sleep hygiene guidance
  - Mindfulness and relaxation practices
  - Lifestyle recommendations for mental wellbeing

IMPORTANT: This agent provides educational guidance ONLY.
It must NOT diagnose mental health conditions or replace
professional psychological or psychiatric care.

Emergency handling: If crisis indicators are detected, the agent
immediately redirects to emergency/crisis resources.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MENTAL_HEALTH_SYSTEM_PROMPT = """You are the Mental Health Agent of an AI Health Assistant.
You provide compassionate, educational mental wellness guidance.

Core principles:
  - You are NOT a therapist, psychologist, or psychiatrist.
  - NEVER diagnose mental health conditions.
  - ALWAYS recommend professional help for clinical concerns.
  - Be warm, empathetic, non-judgmental, and supportive.

Guidelines:
  - Share evidence-based coping strategies for stress, anxiety, and low mood.
  - Explain sleep hygiene principles clearly.
  - Guide users through basic mindfulness and relaxation techniques.
  - Recommend lifestyle adjustments: exercise, social connection, routine.
  - If the user expresses thoughts of self-harm or suicide, immediately 
    provide crisis resources (988 Suicide & Crisis Lifeline, local emergency services)
    and encourage them to seek immediate help.
  - If the RETRIEVED MEDICAL CONTEXT contains relevant information, use it and cite sources.
  - If not, use your general knowledge and state it explicitly.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED MEDICAL CONTEXT:
{context}
"""

MENTAL_HEALTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MENTAL_HEALTH_SYSTEM_PROMPT),
    ("human", "{question}"),
])

DISCLAIMER = (
    "\n\n---\n"
    "> **🧠 Mental Health Disclaimer:** This information is for educational and "
    "general wellness purposes only. It does not constitute medical advice, diagnosis, "
    "or treatment. If you are experiencing a mental health crisis, please contact the "
    "**988 Suicide & Crisis Lifeline** (call or text **988**) or your local emergency services."
)

CRISIS_KEYWORDS = {
    "suicide", "suicidal", "kill myself", "end my life", "self-harm",
    "self harm", "hurt myself", "not want to live", "want to die",
}


class MentalHealthAgent:
    """
    Provides educational mental wellness guidance.
    """

    def __init__(self, llm):
        self.llm = llm
        self._chain = MENTAL_HEALTH_PROMPT | self.llm | StrOutputParser()

    def _is_crisis(self, query: str) -> bool:
        """Detect potential crisis language in the user query."""
        q = query.lower()
        return any(kw in q for kw in CRISIS_KEYWORDS)

    def run(self, query: str, context: str, chat_history: str) -> str:
        """
        Generate a mental health educational response.

        Args:
            query:        The user's mental health / wellness question.
            context:      Pre-formatted retrieved document chunks (may be empty).
            chat_history: Formatted conversation history string.

        Returns:
            Generated response string with disclaimer appended.
        """
        # Immediate crisis response
        if self._is_crisis(query):
            return (
                "I'm really concerned about what you've shared. Your life has value, "
                "and support is available right now.\n\n"
                "**🆘 Please reach out immediately:**\n"
                "- **988 Suicide & Crisis Lifeline:** Call or text **988** (US)\n"
                "- **Crisis Text Line:** Text HOME to **741741**\n"
                "- **Emergency Services:** Call **911** or go to your nearest ER\n\n"
                "You don't have to face this alone. Please talk to someone who can help."
            )

        try:
            response = self._chain.invoke({
                "question": query,
                "context": context,
                "chat_history": chat_history,
            })
            return response + DISCLAIMER
        except Exception as e:
            return (
                "I'm sorry, I encountered an issue generating a wellness response. "
                f"Please try again. (Error: {e})"
            )
