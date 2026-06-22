"""
Retriever Module
────────────────
Creates a retriever from the vector store that performs
similarity search to find the most relevant document chunks
for a given query.
"""

from langchain_chroma import Chroma
import config


def get_retriever(vectorstore: Chroma, top_k: int = None):
    """
    Create a retriever from the vector store.

    The retriever performs similarity search and returns the top-k
    most relevant chunks for a given query.

    Args:
        vectorstore: Chroma vector store instance.
        top_k:       Number of top results to retrieve (default from config).

    Returns:
        A LangChain retriever object.
    """
    top_k = top_k or config.RETRIEVER_TOP_K

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    return retriever


def format_retrieved_docs(docs: list) -> str:
    """
    Format retrieved documents into a single string for prompt injection.

    Args:
        docs: List of Document objects from the retriever.

    Returns:
        Formatted string combining all retrieved chunks.
    """
    if not docs:
        return "No relevant medical documents found."

    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        # Extract just the filename from the full path
        source_name = source.split("\\")[-1].split("/")[-1]
        formatted_parts.append(
            f"[Source {i}: {source_name}, Page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted_parts)
