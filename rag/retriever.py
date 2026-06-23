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


def search_with_scores(vectorstore: Chroma, query: str, top_k: int = None, filter_dict: dict = None):
    """
    Search the vector store and return chunks with their similarity scores.
    """
    top_k = top_k or config.RETRIEVER_TOP_K
    
    # Chroma returns list of tuples: (Document, score)
    # The score is typically L2 distance (lower is better) or cosine distance depending on config,
    # but langchain's similarity_search_with_score returns the raw score.
    # To be clear, we will just return the raw score for now.
    results = vectorstore.similarity_search_with_score(
        query,
        k=top_k,
        filter=filter_dict
    )
    
    return results

def format_retrieved_docs(results: list) -> str:
    """
    Format retrieved documents (with scores) into a single string for prompt injection.
    Uses professional source attribution with Organization, Document, and Page.
    """
    if not results:
        return "No relevant medical documents found."

    formatted_parts = []
    for doc, _ in results:
        # Extract metadata
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        
        # Determine organization from metadata, fallback to "General"
        org = doc.metadata.get("organization", "")
        if not org or org == "General":
            # Try to infer from the source/filename
            source_upper = source.upper()
            for keyword in ["WHO", "CDC", "NIH", "FDA", "AHA"]:
                if keyword in source_upper:
                    org = keyword
                    break
            else:
                org = "General"
        
        # Determine a clean document title
        title = doc.metadata.get("title", "")
        if not title or title == "Unknown Document" or title == "Unknown":
            # Fallback to filename without extension
            source_name = source.split("\\")[-1].split("/")[-1]
            title = source_name.rsplit(".", 1)[0] if "." in source_name else source_name
            
        formatted_parts.append(
            f"Organization: {org}\nDocument: {title}\nPage: {page}\n\n{doc.page_content}"
        )

    return "\n\n--------------------------------\n\n".join(formatted_parts)
