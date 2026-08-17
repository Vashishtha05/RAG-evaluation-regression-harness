from pathlib import Path

from rag_eval.engine import evaluate_files

ROOT = Path(__file__).parents[1]


def test_engine_keeps_retrieval_and_generation_scores_separate() -> None:
    result = evaluate_files(ROOT / "data" / "golden.json", ROOT / "data" / "corpus" / "constitution_chunks.json")

    assert "retrieval_precision" in result.aggregate_scores
    assert "faithfulness" in result.aggregate_scores
    assert "abstention" in result.aggregate_scores

    # Answerable cases carry retrieval metrics but no abstention score.
    answerable = result.query_scores["constitution-001"]
    assert "retrieval_precision" in answerable
    assert "abstention" not in answerable

    # Unanswerable cases carry an abstention score and no retrieval metrics.
    unanswerable = result.query_scores["constitution-021"]
    assert "abstention" in unanswerable
    assert not any(key.startswith("retrieval_") for key in unanswerable)

    assert result.query_artifacts["constitution-001"]["retrieved_chunk_ids"]