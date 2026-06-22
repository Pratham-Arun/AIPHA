"""
Document Loader Module
──────────────────────
Loads medical PDF documents from the data/medical_docs directory.
Uses PyPDFLoader to parse PDF files into LangChain Document objects.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_single_pdf(file_path: str | Path) -> list:
    """
    Load a single PDF file and return a list of Document objects.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects, one per page.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")

    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    print(f"  Loaded '{file_path.name}': {len(documents)} page(s)")
    return documents


def load_all_pdfs(directory: str | Path) -> list:
    """
    Load all PDF files from a directory.

    Args:
        directory: Path to the directory containing PDF files.

    Returns:
        List of Document objects from all PDFs.
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"  Directory does not exist: {directory}")
        return []

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        print(f"  No PDF files found in: {directory}")
        return []

    print(f"  Found {len(pdf_files)} PDF file(s) in {directory.name}/")

    all_documents = []
    for pdf_file in pdf_files:
        try:
            docs = load_single_pdf(pdf_file)
            all_documents.extend(docs)
        except Exception as e:
            print(f"  Error loading {pdf_file.name}: {e}")

    print(f"  Total pages loaded: {len(all_documents)}")
    return all_documents
