"""Generation scorers and calibration helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .retrieval import HashEmbeddingProvider, tokenize


class Judge(Protocol):
    model_id: str

    def faithfulness(self, answer: str, context: str) -> float: ...
    def relevance(self, query: str, answer: str) -> float: ...
    def correctness(self, answer: str, expected: str) -> float: ...


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]


def _overlap(left: str, right: str) -> float:
    left_tokens, right_tokens = tokenize(left), tokenize(right)
    return len(left_tokens & right_tokens) / len(left_tokens) if left_tokens else 0.0


# Phrases that signal the system declined to answer from the corpus. The default
# offline answerer emits "I do not know from the provided corpus." and the golden
# abstention answers use the same register.
ABSTENTION_MARKERS = (
    "do not know",
    "don't know",
    "cannot answer",
    "can't answer",
    "cannot be answered",
    "not answerable",
    "unable to answer",
    "does not specify",
    "does not name",
    "does not mention",
    "not specified",
    "no information",
    "insufficient",
)


def abstained(answer: str) -> float:
    """Return 1.0 when an answer declines to answer from the corpus, else 0.0.

    This is the correct behavior for unanswerable questions; scoring it keeps
    abstention out of the retrieval averages while still holding the pipeline
    accountable for over-answering.
    """
    text = answer.lower()
    return 1.0 if any(marker in text for marker in ABSTENTION_MARKERS) else 0.0


@dataclass(frozen=True)
class HeuristicJudge:
    model_id: str = "heuristic-v1"

    def faithfulness(self, answer: str, context: str) -> float:
        claims = _sentences(answer)
        return sum(_overlap(claim, context) >= 0.5 for claim in claims) / len(claims) if claims else 0.0

    def relevance(self, query: str, answer: str) -> float:
        return min(1.0, _overlap(query, answer) * 2)

    def correctness(self, answer: str, expected: str) -> float:
        return (_overlap(answer, expected) + _overlap(expected, answer)) / 2


class OpenAIJudge:
    """Pinned LLM judge; the client is injected to keep tests and providers swappable."""

    def __init__(self, client, model: str = "gpt-4o-mini", rubric_version: str = "judge-rubric-v1") -> None:
        self.client = client
        self.model_id = f"openai:{model}:{rubric_version}"

    def _score(self, instruction: str) -> float:
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id.split(":")[1],
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "Return only JSON with a numeric score from 0 to 1."},
                        {"role": "user", "content": instruction},
                    ],
                )
                score = float(json.loads(response.choices[0].message.content)["score"])
                if not 0 <= score <= 1:
                    raise ValueError("LLM judge score must be between 0 and 1")
                return score
            except Exception as error:  # noqa: BLE001 - provider SDK errors vary by transport
                last_error = error
        raise RuntimeError("OpenAI judge failed after 3 attempts") from last_error

    def faithfulness(self, answer: str, context: str) -> float:
        return self._score(f"Score whether every claim in ANSWER is supported by CONTEXT.\nANSWER: {answer}\nCONTEXT: {context}")

    def relevance(self, query: str, answer: str) -> float:
        return self._score(f"Score whether ANSWER directly addresses QUESTION.\nQUESTION: {query}\nANSWER: {answer}")

    def correctness(self, answer: str, expected: str) -> float:
        return self._score(f"Score whether ANSWER is correct compared with EXPECTED.\nANSWER: {answer}\nEXPECTED: {expected}")


class HybridJudge:
    """Use embedding similarity and an LLM rubric for correctness."""

    def __init__(self, llm_judge: Judge, semantic_weight: float = 0.4) -> None:
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        self.llm_judge = llm_judge
        self.semantic_weight = semantic_weight
        self.model_id = f"hybrid:{llm_judge.model_id}:semantic-{semantic_weight:.2f}"
        self.embeddings = HashEmbeddingProvider()

    def faithfulness(self, answer: str, context: str) -> float:
        return self.llm_judge.faithfulness(answer, context)

    def relevance(self, query: str, answer: str) -> float:
        return self.llm_judge.relevance(query, answer)

    def correctness(self, answer: str, expected: str) -> float:
        answer_vector = self.embeddings.embed(answer)
        expected_vector = self.embeddings.embed(expected)
        semantic = sum(left * right for left, right in zip(answer_vector, expected_vector))
        llm_score = self.llm_judge.correctness(answer, expected)
        return self.semantic_weight * semantic + (1 - self.semantic_weight) * llm_score


@dataclass(frozen=True)
class CalibrationResult:
    agreement: float
    sample_size: int
    judge_model: str


def calibrate(labels: list[bool], scores: list[float], judge_model: str) -> CalibrationResult:
    if len(labels) != len(scores) or not labels:
        raise ValueError("Calibration requires equally sized, non-empty labels and scores")
    predictions = [score >= 0.5 for score in scores]
    return CalibrationResult(sum(a == b for a, b in zip(labels, predictions)) / len(labels), len(labels), judge_model)