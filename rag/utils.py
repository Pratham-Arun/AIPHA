"""Shared helpers for RAG chunk identity."""


def make_chunk_id(doc_id: str, page: int, chunk_number: int) -> str:
    """Build a deterministic ChromaDB chunk ID."""
    return f"{doc_id}_page{page}_chunk{chunk_number}"


def assign_chunk_ids(chunks: list, doc_id: str) -> list[str]:
    """
    Assign deterministic IDs to chunks and set chunk_number in metadata.

    Returns:
        List of chunk IDs in the same order as chunks.
    """
    page_chunk_counters: dict[int, int] = {}
    ids: list[str] = []

    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        page_chunk_counters[page] = page_chunk_counters.get(page, 0) + 1
        chunk_number = page_chunk_counters[page]
        chunk.metadata["chunk_number"] = chunk_number
        ids.append(make_chunk_id(str(doc_id), page, chunk_number))

    return ids
