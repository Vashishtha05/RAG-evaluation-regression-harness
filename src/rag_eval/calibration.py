"""Run pinned judge models against human-labeled calibration examples."""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import GoldenCase
from .judges import CalibrationResult, Judge, calibrate
from .pipeline import RAGPipeline


def load_labels(path: str | Path) -> dict[str, dict[str, bool]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {record["case_id"]: record for record in records}


def calibrate_pipeline(
    cases: list[GoldenCase], labels: dict[str, dict[str, bool]], pipeline: RAGPipeline, judge: Judge
) -> dict[str, CalibrationResult]:
    faithfulness_labels: list[bool] = []
    faithfulness_scores: list[float] = []
    relevance_labels: list[bool] = []
    relevance_scores: list[float] = []
    cases_by_id = {case.case_id: case for case in cases}
    for case_id, human in labels.items():
        case = cases_by_id.get(case_id)
        query = human.get("query", case.query if case else "")
        if not query:
            continue
        if "answer" in human and "context" in human:
            answer = human["answer"]
            context = human["context"]
        else:
            result = pipeline.run(query)
            answer = result.answer
            context = " ".join(result.context)
        faithfulness_labels.append(human["faithful"])
        faithfulness_scores.append(judge.faithfulness(answer, context))
        relevance_labels.append(human["relevant"])
        relevance_scores.append(judge.relevance(query, answer))
    return {
        "faithfulness": calibrate(faithfulness_labels, faithfulness_scores, judge.model_id),
        "relevance": calibrate(relevance_labels, relevance_scores, judge.model_id),
    }