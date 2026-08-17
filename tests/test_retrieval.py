from rag_eval.config import RunConfig
from rag_eval.retrieval import (
    Chunk,
    HashEmbeddingProvider,
    KeywordRetriever,
    VectorRetriever,
    chunk_text,
)
from rag_eval.scorers import mean_reciprocal_rank, precision_at_k, recall_at_k, score_retrieval


def test_retrieval_metrics_are_deterministic() -> None:
    retrieved = ["a", "wrong", "b"]
    relevant = {"a", "b"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert mean_reciprocal_rank(retrieved, relevant) == 1.0
    assert score_retrieval(retrieved, relevant, 3).recall_at_k == 1.0


def test_config_hash_changes_when_a_knob_changes() -> None:
    assert RunConfig(top_k=3).hash() != RunConfig(top_k=4).hash()


def test_chunking_honors_overlap_and_pipeline_retrieves_top_k() -> None:
    chunks = chunk_text("doc", "one two three four five six", size=4, overlap=1)
    pipeline_chunks = [Chunk("a", "cats chase mice"), Chunk("b", "birds fly"), Chunk("c", "cats sleep")]

    assert len(chunks) == 2
    assert chunk_text("short", "one two", size=4, overlap=1)[0].chunk_id == "short"
    assert chunks[0].text.endswith("four")
    assert chunks[1].text.startswith("four")
    assert len(KeywordRetriever(pipeline_chunks).search("cats", 2)) == 2


def test_vector_retriever_is_injectable_and_deterministic() -> None:
    chunks = [Chunk("a", "cats chase mice"), Chunk("b", "birds fly")]
    provider = HashEmbeddingProvider(dimensions=32)
    retriever = VectorRetriever(chunks, provider)

    assert provider.embed("cats") == provider.embed("cats")
    assert retriever.search("cats", 1)[0].chunk_id == "a"