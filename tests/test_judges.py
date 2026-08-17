from types import SimpleNamespace

import pytest

from rag_eval.cli import build_judge
from rag_eval.judges import HeuristicJudge, HybridJudge, OpenAIJudge, abstained
from rag_eval.pipeline import OpenAIAnswerer
from rag_eval.retrieval import Chunk


def test_abstained_detects_declining_answers() -> None:
    assert abstained("I do not know from the provided corpus.") == 1.0
    assert abstained("The corpus does not specify a federal voting age.") == 1.0
    assert abstained("The Senate is composed of two Senators from each state.") == 0.0


def test_build_judge_resolves_specs() -> None:
    assert isinstance(build_judge("heuristic"), HeuristicJudge)
    with pytest.raises(ValueError, match="Unknown judge"):
        build_judge("mystery-judge")


class FakeChat:
    def __init__(self, content: str) -> None:
        self.calls = 0
        self.content = content

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChat("answer"))


def test_openai_answerer_uses_prompt_template_and_cache() -> None:
    client = FakeClient()
    answerer = OpenAIAnswerer(client, prompt_template="concise-grounded-v1")

    assert answerer.answer("question", [Chunk("one", "context")]) == "answer"
    assert answerer.answer("question", [Chunk("one", "context")]) == "answer"
    assert client.chat.completions.calls == 1


def test_hybrid_judge_combines_llm_score_with_embedding_similarity() -> None:
    client = FakeClient()
    client.chat.completions.content = '{"score": 0.8}'
    judge = HybridJudge(OpenAIJudge(client))

    score = judge.correctness("The Senate has two senators.", "The Senate has two senators.")

    assert 0.4 < score <= 1.0