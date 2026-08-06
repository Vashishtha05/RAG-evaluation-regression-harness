"""SQLite persistence for complete, traceable evaluation runs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    config_hash: str
    git_commit: str
    aggregate_scores: dict[str, float]
    created_at: str


class ResultsStore:
    def __init__(self, path: str = "results.sqlite3") -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, config_hash TEXT NOT NULL, git_commit TEXT NOT NULL, aggregate_scores TEXT NOT NULL, created_at TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS query_scores (run_id TEXT NOT NULL, case_id TEXT NOT NULL, scores TEXT NOT NULL, PRIMARY KEY (run_id, case_id))")
        self.connection.execute("CREATE TABLE IF NOT EXISTS query_artifacts (run_id TEXT NOT NULL, case_id TEXT NOT NULL, artifacts TEXT NOT NULL, PRIMARY KEY (run_id, case_id))")
        self.connection.commit()

    def save_run(
        self,
        summary: RunSummary,
        query_scores: dict[str, dict[str, float]],
        query_artifacts: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.connection.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)", (summary.run_id, summary.config_hash, summary.git_commit, json.dumps(summary.aggregate_scores), summary.created_at))
        self.connection.executemany("INSERT OR REPLACE INTO query_scores VALUES (?, ?, ?)", [(summary.run_id, case_id, json.dumps(scores)) for case_id, scores in query_scores.items()])
        if query_artifacts:
            self.connection.executemany("INSERT OR REPLACE INTO query_artifacts VALUES (?, ?, ?)", [(summary.run_id, case_id, json.dumps(artifacts)) for case_id, artifacts in query_artifacts.items()])
        self.connection.commit()

    def get_run(self, run_id: str) -> RunSummary:
        row = self.connection.execute("SELECT run_id, config_hash, git_commit, aggregate_scores, created_at FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunSummary(row[0], row[1], row[2], json.loads(row[3]), row[4])

    def list_runs(self) -> list[RunSummary]:
        rows = self.connection.execute(
            "SELECT run_id, config_hash, git_commit, aggregate_scores, created_at "
            "FROM runs ORDER BY created_at DESC"
        ).fetchall()
        return [RunSummary(row[0], row[1], row[2], json.loads(row[3]), row[4]) for row in rows]

    def get_query_scores(self, run_id: str) -> dict[str, dict[str, float]]:
        rows = self.connection.execute(
            "SELECT case_id, scores FROM query_scores WHERE run_id = ? ORDER BY case_id", (run_id,)
        ).fetchall()
        return {case_id: json.loads(scores) for case_id, scores in rows}

    def get_query_artifacts(self, run_id: str) -> dict[str, dict[str, object]]:
        rows = self.connection.execute(
            "SELECT case_id, artifacts FROM query_artifacts WHERE run_id = ? ORDER BY case_id", (run_id,)
        ).fetchall()
        return {case_id: json.loads(artifacts) for case_id, artifacts in rows}

    @staticmethod
    def new_summary(run_id: str, config_hash: str, git_commit: str, aggregate_scores: dict[str, float]) -> RunSummary:
        return RunSummary(run_id, config_hash, git_commit, aggregate_scores, datetime.now(UTC).isoformat())