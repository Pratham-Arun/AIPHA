Place medical PDF documents in this directory.

The RAG pipeline will automatically:
1. Load all .pdf files from this folder
2. Split them into chunks
3. Generate embeddings using Gemini
4. Store vectors in ChromaDB

Supported file types: .pdf

Example documents you could add:
- WHO guidelines on nutrition
- Medical reference documents
- Health guidelines from trusted sources
