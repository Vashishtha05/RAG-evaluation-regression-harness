"""Deterministic retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalScores:
    precision_at_k: float
    recall_at_k: float
    mrr: float


def precision_at_k(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    selected = retrieved[:k] if k is not None else retrieved
    return sum(chunk_id in relevant for chunk_id in selected) / len(selected) if selected else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    selected = retrieved[:k] if k is not None else retrieved
    return sum(chunk_id in relevant for chunk_id in selected) / len(relevant) if relevant else 1.0


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def score_retrieval(retrieved: list[str], relevant: set[str], k: int) -> RetrievalScores:
    return RetrievalScores(
        precision_at_k(retrieved, relevant, k),
        recall_at_k(retrieved, relevant, k),
        mean_reciprocal_rank(retrieved, relevant),
    )