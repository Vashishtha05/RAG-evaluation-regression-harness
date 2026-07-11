"""Optional Streamlit dashboard for stored evaluation runs."""

from pathlib import Path

import streamlit as st

from rag_eval.regression import compare_runs
from rag_eval.store import ResultsStore

st.set_page_config(page_title="RAG Quality", layout="wide")
st.title("RAG quality")
st.caption("Retrieval and generation quality stay separate so regressions are diagnosable.")

database = st.text_input("Results database", "results.sqlite3")
if not Path(database).exists():
    st.info("Run an evaluation to create a results database.")
else:
    store = ResultsStore(database)
    runs = store.list_runs()
    if not runs:
        st.info("No evaluation runs have been saved yet.")
    else:
        history = {metric: [run.aggregate_scores.get(metric, 0.0) for run in reversed(runs)] for metric in runs[0].aggregate_scores}
        st.subheader("Scores over time")
        st.line_chart(history)
        selected_id = st.selectbox("Run", [run.run_id for run in runs])
        selected = store.get_run(selected_id)
        st.caption(f"Config {selected.config_hash} | commit {selected.git_commit} | {selected.created_at}")
        st.subheader("Aggregate scores")
        # A dict of scalars cannot be turned into a DataFrame directly ("If using
        # all scalar values, you must pass an index"); render one row per metric.
        st.dataframe(
            [{"metric": metric, "score": score} for metric, score in selected.aggregate_scores.items()],
            hide_index=True,
            use_container_width=True,
        )

        other_ids = [run.run_id for run in runs if run.run_id != selected_id]
        if other_ids:
            st.subheader("Current vs. baseline")
            baseline_id = st.selectbox("Baseline run", other_ids)
            threshold = st.number_input(
                "Regression threshold (per metric)", min_value=0.0, max_value=1.0, value=0.05, step=0.01
            )
            baseline_run = store.get_run(baseline_id)
            metrics = sorted(set(selected.aggregate_scores) | set(baseline_run.aggregate_scores))
            report = compare_runs(
                baseline_run.aggregate_scores,
                selected.aggregate_scores,
                {metric: threshold for metric in metrics},
            )
            st.dataframe(
                [
                    {
                        "metric": diff.metric,
                        "baseline": round(diff.baseline, 3),
                        "current": round(diff.candidate, 3),
                        "delta": round(diff.delta, 3),
                        "status": "REGRESSED" if diff.regressed else "ok",
                    }
                    for diff in report.diffs
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"Gate: {'PASS' if report.passed else 'FAIL'} against `{baseline_id}`")

        query_scores = store.get_query_scores(selected_id)
        query_artifacts = store.get_query_artifacts(selected_id)
        if query_scores:
            st.subheader("Query drill-down")
            view = st.radio("Failure type", ["All", "Retrieval", "Generation"], horizontal=True)
            if view == "Retrieval":
                metric_prefix = "retrieval_"
                case_ids = [case_id for case_id, scores in query_scores.items() if any(key.startswith(metric_prefix) and value < 0.5 for key, value in scores.items())]
            elif view == "Generation":
                case_ids = [case_id for case_id, scores in query_scores.items() if any(not key.startswith("retrieval_") and value < 0.5 for key, value in scores.items())]
            else:
                case_ids = list(query_scores)
            if not case_ids:
                st.info("No cases match this filter.")
                st.stop()
            selected_case = st.selectbox("Case", case_ids)
            scores = query_scores[selected_case]
            retrieval = {key: value for key, value in scores.items() if key.startswith("retrieval_")}
            generation = {key: value for key, value in scores.items() if not key.startswith("retrieval_")}
            left, right = st.columns(2)
            # Unanswerable cases carry an abstention score instead of retrieval metrics.
            if retrieval:
                left.metric("Retrieval precision", f"{retrieval['retrieval_precision']:.3f}")
                left.json(retrieval)
            elif "abstention" in generation:
                left.metric("Abstention", f"{generation['abstention']:.3f}")
            right.metric("Correctness", f"{generation.get('correctness', 0.0):.3f}")
            right.json(generation)
            artifact = query_artifacts.get(selected_case, {})
            st.write("**Question**", artifact.get("query", ""))
            st.write("**Retrieved chunks**", artifact.get("retrieved_chunk_ids", []))
            st.write("**Context**", artifact.get("context", []))
            st.write("**Answer**", artifact.get("answer", ""))