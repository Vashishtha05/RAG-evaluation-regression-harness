"""The intentionally simple RAG system under test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import RunConfig
from .retrieval import Chunk, KeywordRetriever


class Answerer(Protocol):
    def answer(self, query: str, context: list[Chunk]) -> str: ...


class ContextAnswerer:
    """Offline baseline answerer used to exercise retrieval independently."""

    def answer(self, query: str, context: list[Chunk]) -> str:
        if not context:
            return "I do not know from the provided corpus."
        return " ".join(chunk.text for chunk in context)


class OpenAIAnswerer:
    """Optional generation adapter with context-first prompting."""

    def __init__(self, client, model: str = "gpt-4o-mini", prompt_template: str = "context-first-v1") -> None:
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self._cache: dict[tuple[str, str, str], str] = {}

    def _messages(self, query: str, source: str) -> list[dict[str, str]]:
        templates = {
            "context-first-v1": (
                "Answer only from the supplied context. If it is insufficient, say you do not know from the provided corpus.",
                f"Context:\n{source}\n\nQuestion: {query}",
            ),
            "concise-grounded-v1": (
                "Give a concise answer grounded only in the supplied context. Abstain when the context is insufficient.",
                f"Question: {query}\n\nEvidence:\n{source}",
            ),
        }
        try:
            system, user = templates[self.prompt_template]
        except KeyError as error:
            raise ValueError(f"Unknown prompt template: {self.prompt_template}") from error
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def answer(self, query: str, context: list[Chunk]) -> str:
        source = "\n\n".join(chunk.text for chunk in context)
        key = (self.model, self.prompt_template, query + "\n" + source)
        if key in self._cache:
            return self._cache[key]
        error = None
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, temperature=0, messages=self._messages(query, source)
                )
                answer = response.choices[0].message.content.strip()
                self._cache[key] = answer
                return answer
            except Exception as caught:  # noqa: BLE001 - provider SDK errors vary by transport
                error = caught
        raise RuntimeError("OpenAI answer generation failed after 3 attempts") from error


@dataclass(frozen=True)
class PipelineResult:
    query: str
    retrieved_chunk_ids: tuple[str, ...]
    context: tuple[str, ...]
    answer: str


class RAGPipeline:
    def __init__(
        self,
        chunks: list[Chunk],
        config: RunConfig,
        answerer: Answerer | None = None,
        retriever=None,
    ) -> None:
        self.config = config
        self.retriever = retriever or KeywordRetriever(chunks)
        self.answerer = answerer or ContextAnswerer()

    def run(self, query: str) -> PipelineResult:
        retrieved = self.retriever.search(query, self.config.top_k)
        return PipelineResult(
            query=query,
            retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in retrieved),
            context=tuple(chunk.text for chunk in retrieved),
            answer=self.answerer.answer(query, retrieved),
        )