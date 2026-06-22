from langchain_core.prompts import ChatPromptTemplate
from .system_prompt import SYSTEM_PROMPT

# ── Phase 3 & 4 Prompt Template ──
# The prompt now includes three variables:
#   {chat_history} – injected from MemoryManager
#   {context}       – injected from RAG pipeline (retrieved chunks)
#   {question}      – the user's current question
health_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])
