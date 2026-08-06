"""Small, swappable retrieval primitives used by the default pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from math import sqrt
from typing import Protocol

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str


class EmbeddingProvider(Protocol):
    model_id: str

    def embed(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Dependency-free embedding baseline with stable dimensions and model identity."""

    model_id = "hash-embedding-v1"

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            vector[index] += 1.0
        magnitude = sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class OpenAIEmbeddingProvider:
    """OpenAI embedding adapter with a pinned model identity."""

    def __init__(self, client, model: str = "text-embedding-3-small") -> None:
        self.client = client
        self.model_id = f"openai:{model}"

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_id.split(":", 1)[1], input=text)
        return list(response.data[0].embedding)


def tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def chunk_text(document_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [Chunk(document_id, text)]
    step = size - overlap
    return [
        Chunk(f"{document_id}-{start // step:04d}", " ".join(words[start : start + size]))
        for start in range(0, len(words), step)
        if words[start : start + size]
    ]


class KeywordRetriever:
    """Deterministic retriever; replaceable with FAISS or Chroma behind this API."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = tuple(chunks)

    def search(self, query: str, top_k: int) -> list[Chunk]:
        query_tokens = tokenize(query)
        ranked = sorted(
            self._chunks,
            key=lambda chunk: (len(query_tokens & tokenize(chunk.text)), chunk.chunk_id),
            reverse=True,
        )
        return ranked[:top_k]


class VectorRetriever:
    """In-memory cosine retriever; its provider can be replaced with a hosted embedder."""

    def __init__(self, chunks: list[Chunk], embeddings: EmbeddingProvider | None = None) -> None:
        self._chunks = tuple(chunks)
        self._embeddings = embeddings or HashEmbeddingProvider()
        self._vectors = tuple(self._embeddings.embed(chunk.text) for chunk in chunks)

    def search(self, query: str, top_k: int) -> list[Chunk]:
        query_vector = self._embeddings.embed(query)
        ranked = sorted(
            zip(self._chunks, self._vectors),
            key=lambda item: (sum(left * right for left, right in zip(query_vector, item[1])), item[0].chunk_id),
            reverse=True,
        )
        return [chunk for chunk, _ in ranked[:top_k]]


class FaissRetriever(VectorRetriever):
    """Optional FAISS-backed retriever with the same search contract."""

    def __init__(self, chunks: list[Chunk], embeddings: EmbeddingProvider | None = None) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Install the vector extra to use FaissRetriever") from error
        super().__init__(chunks, embeddings)
        self._faiss = faiss
        self._numpy = np
        matrix = np.asarray(self._vectors, dtype="float32")
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    def search(self, query: str, top_k: int) -> list[Chunk]:
        vector = self._numpy.asarray([self._embeddings.embed(query)], dtype="float32")
        _, indices = self._index.search(vector, min(top_k, len(self._chunks)))
        return [self._chunks[index] for index in indices[0] if index >= 0]