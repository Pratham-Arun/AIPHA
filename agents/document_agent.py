"""
Document QA Agent  –  Phase 7 / 7.1
──────────────────────────────────────
Handles medical questions grounded in the uploaded document corpus.

Phase 7.1 enhancement:
  Accepts an optional *doc_analysis_output* string produced by
  DocumentAnalysisTool.  When present, the structural analysis (page count,
  doc type, key findings) is injected alongside the RAG context so the LLM
  can produce a richer, more specific explanation.

Responsibilities:
  - Retrieve relevant chunks from ChromaDB (via the Retrieval Tool)
  - Generate answers with proper source citations
  - Acknowledge when documents do not contain the answer and fall back
    to general medical knowledge with an explicit disclaimer
  - Explain document analysis results in plain language (Phase 7.1)
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DOCUMENT_QA_SYSTEM_PROMPT = """You are the Document QA Agent of an AI Health Assistant.
You specialise in answering medical questions using the provided document corpus.

Guidelines:
  - If the RETRIEVED MEDICAL CONTEXT contains the answer, answer ONLY from that context
    and cite every source (Organization, Document, Page).
  - If the context does NOT contain the answer, clearly state:
    "The uploaded medical documents do not contain this information."
    Then provide a concise answer from your general medical knowledge and explicitly
    note it comes from general knowledge.
  - If DOCUMENT ANALYSIS RESULTS are provided, use them to explain the document
    structure and key findings in plain language. Do NOT re-derive these values.
  - Never diagnose diseases or replace professional medical advice.
  - Be precise, empathetic, and use plain language.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED MEDICAL CONTEXT:
{context}

DOCUMENT ANALYSIS RESULTS (structural analysis — use as-is):
{doc_analysis_output}
"""

DOCUMENT_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DOCUMENT_QA_SYSTEM_PROMPT),
    ("human", "{question}"),
])


class DocumentQAAgent:
    """
    Generates document-grounded answers using the RAG pipeline.
    In Phase 7.1, can also explain document analysis results produced by
    DocumentAnalysisTool.
    """

    def __init__(self, llm):
        self.llm = llm
        self._chain = DOCUMENT_QA_PROMPT | self.llm | StrOutputParser()

    def run(
        self,
        query: str,
        context: str,
        chat_history: str,
        doc_analysis_output: str = "",
    ) -> str:
        """
        Generate a response grounded in retrieved medical context.

        Args:
            query:               The user's question.
            context:             Pre-formatted retrieved document chunks.
            chat_history:        Formatted conversation history string.
            doc_analysis_output: Pre-computed analysis string from DocumentAnalysisTool
                                 (empty string if not analysing a specific document).

        Returns:
            Generated response string.
        """
        analysis_section = (
            doc_analysis_output if doc_analysis_output
            else "No document analysis performed."
        )
        try:
            return self._chain.invoke({
                "question":            query,
                "context":             context,
                "chat_history":        chat_history,
                "doc_analysis_output": analysis_section,
            })
        except Exception as e:
            return (
                "I'm sorry, I encountered an issue retrieving information from the "
                f"medical documents. Please try again. (Error: {e})"
            )
