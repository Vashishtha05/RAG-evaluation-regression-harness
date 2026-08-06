"""Golden dataset models and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]
    expected_answer: str
    source_context: tuple[str, ...]
    answerable: bool
    verified: bool
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GoldenCase:
        required = {
            "case_id",
            "query",
            "relevant_chunk_ids",
            "expected_answer",
            "source_context",
            "answerable",
            "verified",
            "tags",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"Golden case {raw.get('case_id', '<unknown>')} is missing: {sorted(missing)}")
        return cls(
            case_id=str(raw["case_id"]),
            query=str(raw["query"]),
            relevant_chunk_ids=tuple(str(value) for value in raw["relevant_chunk_ids"]),
            expected_answer=str(raw["expected_answer"]),
            source_context=tuple(str(value) for value in raw["source_context"]),
            answerable=bool(raw["answerable"]),
            verified=bool(raw["verified"]),
            tags=tuple(str(value) for value in raw["tags"]),
        )


def load_golden_dataset(path: str | Path) -> list[GoldenCase]:
    """Load and validate the versioned golden dataset."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Golden dataset must be a JSON array")
    cases = [GoldenCase.from_dict(item) for item in payload]
    validate_golden_dataset(cases)
    return cases


def validate_golden_dataset(cases: list[GoldenCase]) -> None:
    """Reject ambiguous labels before they can affect an evaluation run."""
    if not cases:
        raise ValueError("Golden dataset cannot be empty")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Golden case IDs must be unique")
    for case in cases:
        if not case.query.strip():
            raise ValueError(f"{case.case_id}: query cannot be empty")
        if not case.expected_answer.strip():
            raise ValueError(f"{case.case_id}: expected_answer cannot be empty")
        if case.answerable and not case.relevant_chunk_ids:
            raise ValueError(f"{case.case_id}: answerable cases need relevant chunks")
        if not case.answerable and case.relevant_chunk_ids:
            raise ValueError(f"{case.case_id}: unanswerable cases cannot have relevant chunks")
        if len(case.source_context) != len(case.relevant_chunk_ids):
            raise ValueError(f"{case.case_id}: source_context must align with relevant_chunk_ids")
        if not case.verified:
            continue
        if not case.tags:
            raise ValueError(f"{case.case_id}: verified cases need at least one tag")
