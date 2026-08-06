"""Run the golden set and keep retrieval and generation diagnostics separate."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig
from .dataset import GoldenCase, load_golden_dataset
from .judges import HeuristicJudge, HybridJudge, Judge, OpenAIJudge, abstained
from .pipeline import OpenAIAnswerer, RAGPipeline
from .retrieval import (
    Chunk,
    FaissRetriever,
    HashEmbeddingProvider,
    KeywordRetriever,
    OpenAIEmbeddingProvider,
    VectorRetriever,
    chunk_text,
)
from .scorers import score_retrieval


@dataclass(frozen=True)
class EvaluationResult:
    aggregate_scores: dict[str, float]
    query_scores: dict[str, dict[str, float]]
    query_artifacts: dict[str, dict[str, object]]


def load_corpus(path: str | Path, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Chunk]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks = []
    for record in records:
        chunks.extend(chunk_text(record["chunk_id"], record["text"], chunk_size, chunk_overlap))
    return chunks


def warn_on_unknown_chunk_ids(cases: list[GoldenCase], chunks: list[Chunk]) -> set[str]:
    """Warn when golden labels reference chunk IDs the corpus does not produce.

    ``chunk_text`` renames a record's chunks (``id-0000`` ...) whenever the record
    is large enough to split, so a smaller ``chunk_size`` can silently break the
    mapping between ``relevant_chunk_ids`` and the retrievable chunks. Surfacing
    it keeps a misconfiguration from masquerading as a retrieval regression.
    """
    available = {chunk.chunk_id for chunk in chunks}
    referenced = {chunk_id for case in cases for chunk_id in case.relevant_chunk_ids}
    missing = referenced - available
    if missing:
        warnings.warn(
            "Golden dataset references chunk IDs absent from the corpus "
            f"(check chunk_size/chunk_overlap): {sorted(missing)}",
            stacklevel=2,
        )
    return missing


def evaluate(cases: list[GoldenCase], pipeline: RAGPipeline, judge: Judge | None = None) -> EvaluationResult:
    generation_judge = judge or HeuristicJudge()
    query_scores: dict[str, dict[str, float]] = {}
    query_artifacts: dict[str, dict[str, object]] = {}
    for case in cases:
        result = pipeline.run(case.query)
        scores: dict[str, float] = {}
        # Retrieval metrics only mean something when there is a relevant chunk to
        # find. For unanswerable cases the correct behavior is to abstain, so we
        # score abstention instead of folding a degenerate precision/recall into
        # the retrieval averages.
        if case.answerable:
            retrieval = score_retrieval(
                list(result.retrieved_chunk_ids), set(case.relevant_chunk_ids), pipeline.config.top_k
            )
            scores["retrieval_precision"] = retrieval.precision_at_k
            scores["retrieval_recall"] = retrieval.recall_at_k
            scores["retrieval_mrr"] = retrieval.mrr
        else:
            scores["abstention"] = abstained(result.answer)
        # Generation metrics apply to every case, answerable or not.
        scores["faithfulness"] = generation_judge.faithfulness(result.answer, " ".join(result.context))
        scores["answer_relevance"] = generation_judge.relevance(case.query, result.answer)
        scores["correctness"] = generation_judge.correctness(result.answer, case.expected_answer)
        query_scores[case.case_id] = scores
        query_artifacts[case.case_id] = {
            "query": case.query,
            "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
            "context": list(result.context),
            "answer": result.answer,
        }
    # Metric-aware aggregation: average each metric only over the cases that
    # carry it, so retrieval and abstention are not diluted by one another.
    totals: dict[str, list[float]] = {}
    for scores in query_scores.values():
        for name, value in scores.items():
            totals.setdefault(name, []).append(value)
    aggregate = {name: sum(values) / len(values) for name, values in totals.items()}
    return EvaluationResult(aggregate, query_scores, query_artifacts)


def evaluate_files(
    dataset_path: str | Path,
    corpus_path: str | Path,
    config: RunConfig | None = None,
    limit: int | None = None,
    verified_only: bool = False,
) -> EvaluationResult:
    selected_config = config or RunConfig()
    chunks = load_corpus(corpus_path, selected_config.chunk_size, selected_config.chunk_overlap)
    client = None
    if selected_config.llm.startswith("openai:") or selected_config.embedding_model.startswith("openai:"):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install the llm extra to use OpenAI providers") from error
        client = OpenAI()
    embeddings = HashEmbeddingProvider()
    if selected_config.embedding_model.startswith("openai:"):
        embeddings = OpenAIEmbeddingProvider(client, selected_config.embedding_model.split(":", 1)[1])
    if selected_config.vector_store == "faiss":
        retriever = FaissRetriever(chunks, embeddings)
    elif selected_config.vector_store == "vector":
        retriever = VectorRetriever(chunks, embeddings)
    else:
        retriever = KeywordRetriever(chunks)
    judge: Judge = HeuristicJudge()
    answerer = None
    if selected_config.llm.startswith("openai:"):
        model = selected_config.llm.split(":", 1)[1]
        answerer = OpenAIAnswerer(client, model, selected_config.prompt_template)
        judge = HybridJudge(OpenAIJudge(client, model))
    pipeline = RAGPipeline(chunks, selected_config, answerer=answerer, retriever=retriever)
    cases = load_golden_dataset(dataset_path)
    if verified_only:
        cases = [case for case in cases if case.verified]
    if limit is not None:
        cases = cases[:limit]
    warn_on_unknown_chunk_ids(cases, chunks)
    return evaluate(cases, pipeline, judge)