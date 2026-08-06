"""Baseline comparison and quality gates."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDiff:
    metric: str
    baseline: float
    candidate: float
    delta: float
    regressed: bool


@dataclass(frozen=True)
class QueryMetricDiff:
    case_id: str
    metric: str
    category: str
    baseline: float
    candidate: float
    delta: float
    regressed: bool


@dataclass(frozen=True)
class RegressionReport:
    diffs: tuple[MetricDiff, ...]
    query_diffs: tuple[QueryMetricDiff, ...] = ()

    @property
    def passed(self) -> bool:
        return not any(diff.regressed for diff in self.diffs) and not any(
            diff.regressed for diff in self.query_diffs
        )


def compare_runs(baseline: dict[str, float], candidate: dict[str, float], thresholds: dict[str, float]) -> RegressionReport:
    diffs = []
    for metric in sorted(set(baseline) | set(candidate)):
        before, after = baseline.get(metric, 0.0), candidate.get(metric, 0.0)
        delta = after - before
        diffs.append(MetricDiff(metric, before, after, delta, delta < -thresholds.get(metric, 0.0)))
    return RegressionReport(tuple(diffs))


def suggest_thresholds(
    runs: list[dict[str, float]], z: float = 2.0, floor: float = 0.005
) -> dict[str, float]:
    """Derive a per-metric regression threshold from the spread of repeated runs.

    A metric that naturally wobbles between identical-config runs (LLM judges do)
    needs a wider tolerance than a deterministic one, or run-to-run noise trips
    the gate. The threshold is ``z`` sample standard deviations of the metric,
    floored so even a perfectly stable metric keeps a small tolerance. Feed the
    result to ``evaluate --threshold-file`` instead of one flat ``--threshold``.
    """
    if len(runs) < 2:
        raise ValueError("Need at least two runs to estimate metric variance")
    thresholds: dict[str, float] = {}
    for metric in sorted({name for run in runs for name in run}):
        values = [run[metric] for run in runs if metric in run]
        if len(values) < 2:
            thresholds[metric] = floor
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        thresholds[metric] = max(floor, z * math.sqrt(variance))
    return thresholds


def compare_query_scores(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    thresholds: dict[str, float],
) -> tuple[QueryMetricDiff, ...]:
    """Compare every case and classify failures as retrieval or generation regressions."""
    diffs: list[QueryMetricDiff] = []
    for case_id in sorted(set(baseline) | set(candidate)):
        baseline_scores = baseline.get(case_id, {})
        candidate_scores = candidate.get(case_id, {})
        for metric in sorted(set(baseline_scores) | set(candidate_scores)):
            before = baseline_scores.get(metric, 0.0)
            after = candidate_scores.get(metric, 0.0)
            delta = after - before
            category = "retrieval" if metric.startswith("retrieval_") else "generation"
            diffs.append(
                QueryMetricDiff(
                    case_id,
                    metric,
                    category,
                    before,
                    after,
                    delta,
                    delta < -thresholds.get(metric, 0.0),
                )
            )
    return tuple(diffs)