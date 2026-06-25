"""
Retriever Module
────────────────
Creates a retriever from the vector store that performs
similarity search to find the most relevant document chunks
for a given query.
"""

import config
from .vectorstore import VectorStore


def get_retriever(vectorstore: VectorStore, top_k: int = None):
    """
    Create a retriever wrapper around the vector store.

    The retriever performs similarity search and returns the top-k
    most relevant chunks for a given query.
    """
    top_k = top_k or config.RETRIEVER_TOP_K
    return _VectorStoreRetriever(vectorstore, top_k)


class _VectorStoreRetriever:
    """Minimal retriever facade backed by VectorStore.search()."""

    def __init__(self, vectorstore: VectorStore, top_k: int):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def invoke(self, query: str):
        return [doc for doc, _ in self.vectorstore.search(query, k=self.top_k)]


def search_with_scores(
    vectorstore: VectorStore,
    query: str,
    top_k: int = None,
    filter_dict: dict = None,
):
    """Search the vector store and return chunks with their similarity scores."""
    return vectorstore.search(query, k=top_k, filter_dict=filter_dict)


def format_retrieved_docs(results: list) -> str:
    """
    Format retrieved documents (with scores) into a single string for prompt injection.
    Uses professional source attribution with Organization, Document, and Page.
    """
    if not results:
        return "No relevant medical documents found."

    formatted_parts = []
    for doc, _ in results:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")

        org = doc.metadata.get("organization", "")
        if not org or org == "General":
            source_upper = source.upper()
            for keyword in ["WHO", "CDC", "NIH", "FDA", "AHA"]:
                if keyword in source_upper:
                    org = keyword
                    break
            else:
                org = "General"

        title = doc.metadata.get("title", "")
        if not title or title == "Unknown Document" or title == "Unknown":
            source_name = source.split("\\")[-1].split("/")[-1]
            title = source_name.rsplit(".", 1)[0] if "." in source_name else source_name

        formatted_parts.append(
            f"Organization: {org}\nDocument: {title}\nPage: {page}\n\n{doc.page_content}"
        )

    return "\n\n--------------------------------\n\n".join(formatted_parts)
