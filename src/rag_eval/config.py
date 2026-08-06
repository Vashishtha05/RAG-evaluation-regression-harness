"""Traceable configuration for a RAG run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    # Default embedding identity must match the provider actually used by the
    # engine when no OpenAI model is requested (HashEmbeddingProvider.model_id),
    # so the recorded config_hash is faithful to what ran.
    embedding_model: str = "hash-embedding-v1"
    # "in-memory" selects the dependency-free KeywordRetriever; "vector" and
    # "faiss" select the embedding-backed retrievers (see engine.evaluate_files).
    vector_store: str = "in-memory"
    top_k: int = 5
    prompt_template: str = "context-first-v1"
    llm: str = "deterministic-abstaining-v1"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.chunk_overlap < 0:
            raise ValueError("chunk_size must be positive and chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]