from pathlib import Path

from rag_eval.calibration import calibrate_pipeline, load_labels
from rag_eval.config import RunConfig
from rag_eval.dataset import load_golden_dataset
from rag_eval.engine import load_corpus
from rag_eval.judges import HeuristicJudge
from rag_eval.pipeline import RAGPipeline

ROOT = Path(__file__).parents[1]


def test_calibration_uses_versioned_human_labels() -> None:
    cases = load_golden_dataset(ROOT / "data" / "golden.json")
    results = calibrate_pipeline(
        cases,
        load_labels(ROOT / "data" / "calibration.json"),
        RAGPipeline(load_corpus(ROOT / "data" / "corpus" / "constitution_chunks.json"), RunConfig()),
        HeuristicJudge(),
    )

    assert results["faithfulness"].sample_size == 20
    assert results["relevance"].judge_model == "heuristic-v1"