import math

import pytest

from rag_eval.judges import calibrate
from rag_eval.regression import compare_query_scores, compare_runs, suggest_thresholds
from rag_eval.store import ResultsStore


def test_regression_gate_reports_metric_drop() -> None:
    report = compare_runs({"precision": 0.8}, {"precision": 0.7}, {"precision": 0.05})
    assert not report.passed
    assert math.isclose(report.diffs[0].delta, -0.1)


def test_suggest_thresholds_widens_for_noisy_metrics() -> None:
    runs = [
        {"stable": 0.80, "noisy": 0.40},
        {"stable": 0.80, "noisy": 0.60},
        {"stable": 0.80, "noisy": 0.50},
    ]
    thresholds = suggest_thresholds(runs, z=2.0, floor=0.005)

    # A perfectly stable metric only gets the floor; a wobbling one gets a wider band.
    assert thresholds["stable"] == 0.005
    assert thresholds["noisy"] > thresholds["stable"]

    with pytest.raises(ValueError, match="two runs"):
        suggest_thresholds([{"stable": 0.8}])


def test_calibration_is_explicit_about_judge_agreement() -> None:
    result = calibrate([True, False, True, False], [0.9, 0.1, 0.7, 0.8], "judge-v1")
    assert result.agreement == 0.75
    assert result.judge_model == "judge-v1"


def test_results_store_lists_saved_runs(tmp_path) -> None:
    store = ResultsStore(str(tmp_path / "results.sqlite3"))
    summary = store.new_summary("run-1", "config", "commit", {"precision": 0.8})
    store.save_run(summary, {"case-1": {"precision": 0.8}}, {"case-1": {"answer": "answer"}})

    assert [run.run_id for run in store.list_runs()] == ["run-1"]
    assert store.get_query_scores("run-1") == {"case-1": {"precision": 0.8}}
    assert store.get_query_artifacts("run-1") == {"case-1": {"answer": "answer"}}


def test_query_regression_classifies_retrieval_and_generation_failures() -> None:
    diffs = compare_query_scores(
        {"case-1": {"retrieval_recall": 1.0, "correctness": 0.8}},
        {"case-1": {"retrieval_recall": 0.7, "correctness": 0.2}},
        {"retrieval_recall": 0.1, "correctness": 0.1},
    )

    failures = [diff for diff in diffs if diff.regressed]
    assert {(diff.category, diff.metric) for diff in failures} == {
        ("retrieval", "retrieval_recall"),
        ("generation", "correctness"),
    }