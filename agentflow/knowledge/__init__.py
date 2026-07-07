"""Knowledge base management: parsing, chunking, embedding, indexing, and retrieval.

Modules
-------
parser
    Document parsing (PDF, DOCX, TXT, Markdown).
chunking
    Structure-aware chunking strategies (heading-based, code-boundary, paragraph).
embedder
    Embedding interface (``BaseEmbedder``) with TF-IDF and semantic implementations.
index
    ChromaDB vector index for efficient ANN search.
retrieval
    Hybrid retrieval pipeline (vector + lexical search).
store
    High-level ``KnowledgeStore`` tying all components together.
"""
