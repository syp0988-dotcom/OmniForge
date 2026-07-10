"""Offline RAG evaluation toolkit.

Modules
-------
dataset
    JSONL-based evaluation dataset loading, saving, and validation.
metrics
    Retrieval quality metrics: Recall@k, Precision@k, MRR, NDCG@k, Hit Rate.
generator
    LLM-driven question generation from knowledge-base chunks.
runner
    Orchestrator that runs queries against a KnowledgeStore and computes metrics.
"""
