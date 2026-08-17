from pathlib import Path

import pytest

from rag_eval.dataset import GoldenCase, load_golden_dataset, validate_golden_dataset

DATASET = Path(__file__).parents[1] / "data" / "golden.json"


def test_golden_dataset_loads_with_verified_core() -> None:
    cases = load_golden_dataset(DATASET)

    assert len(cases) >= 30
    assert sum(case.verified for case in cases) >= 20
    assert any("unanswerable" in case.tags for case in cases)
    assert any("multi-hop" in case.tags for case in cases)


def test_unanswerable_case_cannot_claim_relevant_chunks() -> None:
    case = GoldenCase(
        case_id="bad",
        query="question",
        relevant_chunk_ids=("chunk-1",),
        expected_answer="I do not know.",
        source_context=("context",),
        answerable=False,
        verified=True,
        tags=("unanswerable",),
    )

    with pytest.raises(ValueError, match="unanswerable cases"):
        validate_golden_dataset([case])