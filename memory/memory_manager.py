"""
Memory Manager Module
─────────────────────
Manages conversation history using LangChain's ChatMessageHistory.
Provides save, load, and clear operations for chat history.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from .session import Session


class MemoryManager:
    """
    Manages conversational memory for the Health Assistant.

    Responsibilities:
        - Maintain chat history (Human + AI messages)
        - Manage user session information
        - Provide history for prompt injection
    """

    def __init__(self, session_id: str = "default"):
        self.session = Session(session_id)
        self.chat_history = ChatMessageHistory()

    def add_user_message(self, message: str) -> None:
        """Store a human message in chat history."""
        self.chat_history.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        """Store an AI message in chat history."""
        self.chat_history.add_ai_message(message)

    def save_conversation(self, user_message: str, ai_message: str) -> None:
        """Save a complete conversation turn (user question + AI response)."""
        self.add_user_message(user_message)
        self.add_ai_message(ai_message)

    def get_chat_history(self) -> list:
        """Return the full list of messages."""
        return self.chat_history.messages

    def get_formatted_history(self, max_messages: int = 20) -> str:
        """
        Return chat history formatted as a string for prompt injection.

        Args:
            max_messages: Maximum number of recent messages to include.
                          Keeps the prompt within token limits.

        Returns:
            A formatted string of the conversation history.
        """
        messages = self.chat_history.messages
        # Keep only the most recent messages to avoid token overflow
        recent = messages[-max_messages:] if len(messages) > max_messages else messages

        if not recent:
            return "No previous conversation."

        formatted_lines = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                formatted_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_lines.append(f"Assistant: {msg.content}")

        return "\n".join(formatted_lines)

    def clear_history(self) -> None:
        """Clear all chat history."""
        self.chat_history.clear()

    def clear_session(self) -> None:
        """Clear user session info."""
        self.session.clear()

    def clear_all(self) -> None:
        """Clear both chat history and session info."""
        self.clear_history()
        self.clear_session()

    def get_message_count(self) -> int:
        """Return the number of messages in history."""
        return len(self.chat_history.messages)
