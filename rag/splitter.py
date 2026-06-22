"""
Text Splitter Module
────────────────────
Splits large documents into smaller chunks for embedding.
Uses RecursiveCharacterTextSplitter which splits on natural boundaries
(paragraphs, sentences, words) to preserve semantic coherence.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def split_documents(documents: list, chunk_size: int = None,
                    chunk_overlap: int = None) -> list:
    """
    Split documents into smaller chunks.

    Args:
        documents:     List of Document objects from the loader.
        chunk_size:    Maximum characters per chunk (default from config).
        chunk_overlap: Number of overlapping characters between chunks
                       (default from config).

    Returns:
        List of smaller Document objects (chunks).
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Split on natural boundaries: double newlines, single newlines,
        # sentences, commas, spaces, then characters
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  Split {len(documents)} page(s) into {len(chunks)} chunk(s)")
    print(f"  Chunk size: {chunk_size} | Overlap: {chunk_overlap}")
    return chunks
