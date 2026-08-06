"""Command-line entry points for the harness."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .calibration import calibrate_pipeline, load_labels
from .config import RunConfig
from .dataset import load_golden_dataset
from .engine import evaluate_files, load_corpus
from .judges import HeuristicJudge, Judge, OpenAIJudge
from .pipeline import RAGPipeline
from .regression import compare_query_scores, compare_runs, suggest_thresholds
from .store import ResultsStore


def build_judge(spec: str) -> Judge:
    """Resolve a --judge spec to a judge instance.

    'heuristic' is deterministic and needs no API key; 'openai:MODEL' calibrates
    the real LLM judge (requires the llm extra and an OpenAI key), which is the
    judge you actually ship, so its human agreement is the number that matters.
    """
    if spec == "heuristic":
        return HeuristicJudge()
    if spec.startswith("openai:"):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install the llm extra to calibrate an OpenAI judge") from error
        return OpenAIJudge(OpenAI(), spec.split(":", 1)[1])
    raise ValueError(f"Unknown judge: {spec!r} (expected 'heuristic' or 'openai:MODEL')")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-dataset")
    validate.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).parents[2] / "data" / "golden.json",
    )
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).parents[2] / "data" / "golden.json"
    )
    evaluate_parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).parents[2] / "data" / "corpus" / "constitution_chunks.json",
    )
    evaluate_parser.add_argument("--store", type=Path, help="SQLite file for saving this run")
    evaluate_parser.add_argument("--run-id", default="latest", help="ID assigned to the saved run")
    evaluate_parser.add_argument("--vector-store", choices=["in-memory", "vector", "faiss"], default="in-memory")
    evaluate_parser.add_argument("--llm", default="deterministic-abstaining-v1", help="deterministic-abstaining-v1 or openai:MODEL")
    evaluate_parser.add_argument("--embedding-model", default="hash-embedding-v1")
    evaluate_parser.add_argument("--prompt-template", default="context-first-v1")
    evaluate_parser.add_argument("--limit", type=int, help="Evaluate only the first N cases")
    evaluate_parser.add_argument("--verified-only", action="store_true", help="Evaluate only human-verified cases")
    evaluate_parser.add_argument("--baseline", help="Saved run ID to compare against")
    evaluate_parser.add_argument("--baseline-file", type=Path, help="JSON file containing baseline aggregate scores")
    evaluate_parser.add_argument("--report-file", type=Path, help="Write a Markdown regression report")
    evaluate_parser.add_argument(
        "--threshold", type=float, default=0.0, help="Allowed drop applied to every metric"
    )
    evaluate_parser.add_argument(
        "--threshold-file",
        type=Path,
        help="JSON of per-metric thresholds (from suggest-thresholds); overrides --threshold per metric",
    )
    thresholds_parser = subparsers.add_parser("suggest-thresholds")
    thresholds_parser.add_argument("--store", type=Path, required=True, help="SQLite file holding the runs")
    thresholds_parser.add_argument("--runs", required=True, help="Comma-separated run IDs to measure variance over")
    thresholds_parser.add_argument("--z", type=float, default=2.0, help="Std-dev multiplier for the tolerance band")
    thresholds_parser.add_argument("--floor", type=float, default=0.005, help="Minimum threshold for a stable metric")
    thresholds_parser.add_argument("--out", type=Path, help="Write the suggested thresholds JSON here")
    calibration_parser = subparsers.add_parser("calibrate")
    calibration_parser.add_argument("--dataset", type=Path, default=Path(__file__).parents[2] / "data" / "golden.json")
    calibration_parser.add_argument("--corpus", type=Path, default=Path(__file__).parents[2] / "data" / "corpus" / "constitution_chunks.json")
    calibration_parser.add_argument("--labels", type=Path, default=Path(__file__).parents[2] / "data" / "calibration.json")
    calibration_parser.add_argument(
        "--judge",
        default="heuristic",
        help="Judge to calibrate: 'heuristic' (deterministic, no API key) or 'openai:MODEL'",
    )
    args = parser.parse_args()
    if args.command == "validate-dataset":
        cases = load_golden_dataset(args.path)
        verified = sum(case.verified for case in cases)
        print(f"Validated {len(cases)} cases ({verified} verified).")
    elif args.command == "evaluate":
        config = RunConfig(
            vector_store=args.vector_store,
            llm=args.llm,
            embedding_model=args.embedding_model,
            prompt_template=args.prompt_template,
        )
        result = evaluate_files(
            args.dataset,
            args.corpus,
            config,
            limit=args.limit,
            verified_only=args.verified_only,
        )
        for metric, score in result.aggregate_scores.items():
            print(f"{metric}: {score:.3f}")
        store = None
        if args.store:
            store = ResultsStore(str(args.store))
            try:
                git_commit = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                git_commit = "unknown"
            summary = store.new_summary(
                args.run_id, config.hash(), git_commit, result.aggregate_scores
            )
            store.save_run(summary, result.query_scores, result.query_artifacts)
            print(f"Saved run {args.run_id} to {args.store}.")
        # The quality gate runs whenever a baseline is supplied, independent of
        # whether the candidate run was persisted with --store. Persistence and
        # gating are separate concerns; coupling them silently disables the gate.
        if args.baseline or args.baseline_file:
            if args.baseline and store is None:
                raise SystemExit(
                    "--baseline <run-id> reads a saved run and therefore requires --store; "
                    "use --baseline-file for a standalone baseline."
                )
            baseline_scores = (
                store.get_run(args.baseline).aggregate_scores
                if args.baseline
                else json.loads(args.baseline_file.read_text(encoding="utf-8"))["aggregate_scores"]
            )
            thresholds = {metric: args.threshold for metric in result.aggregate_scores}
            if args.threshold_file:
                thresholds.update(json.loads(args.threshold_file.read_text(encoding="utf-8")))
            report = compare_runs(baseline_scores, result.aggregate_scores, thresholds)
            if args.baseline:
                report = report.__class__(
                    report.diffs,
                    compare_query_scores(
                        store.get_query_scores(args.baseline), result.query_scores, thresholds
                    ),
                )
            for diff in report.diffs:
                print(f"{diff.metric}: {diff.delta:+.3f}")
            failures = [diff for diff in report.query_diffs if diff.regressed]
            for diff in failures:
                print(
                    f"FAIL {diff.category} {diff.case_id} {diff.metric}: "
                    f"{diff.delta:+.3f}"
                )
            if failures:
                print(f"{len(failures)} per-query regressions detected.")
            if args.report_file:
                lines = ["## RAG quality report", "", "| Metric | Delta | Status |", "| --- | ---: | --- |"]
                lines.extend(
                    f"| {diff.metric} | {diff.delta:+.3f} | {'FAIL' if diff.regressed else 'PASS'} |"
                    for diff in report.diffs
                )
                if failures:
                    lines.extend(["", "### Per-query regressions"])
                    lines.extend(
                        f"- `{diff.category}` `{diff.case_id}` `{diff.metric}`: {diff.delta:+.3f}"
                        for diff in failures
                    )
                lines.extend(["", f"**Gate: {'PASS' if report.passed else 'FAIL'}**"])
                args.report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if not report.passed:
                raise SystemExit(1)
    elif args.command == "suggest-thresholds":
        store = ResultsStore(str(args.store))
        run_ids = [run_id.strip() for run_id in args.runs.split(",") if run_id.strip()]
        runs = [store.get_run(run_id).aggregate_scores for run_id in run_ids]
        thresholds = suggest_thresholds(runs, z=args.z, floor=args.floor)
        for metric, threshold in thresholds.items():
            print(f"{metric}: {threshold:.4f}")
        if args.out:
            args.out.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {len(thresholds)} thresholds to {args.out}.")
    elif args.command == "calibrate":
        cases = load_golden_dataset(args.dataset)
        pipeline = RAGPipeline(load_corpus(args.corpus), RunConfig())
        judge = build_judge(args.judge)
        results = calibrate_pipeline(cases, load_labels(args.labels), pipeline, judge)
        for metric, result in results.items():
            print(f"{metric}: agreement={result.agreement:.3f} sample_size={result.sample_size} judge={result.judge_model}")


if __name__ == "__main__":
    main()