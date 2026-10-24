"""Unit tests for reranker.CrossEncoderReranker.

Only the neural CrossEncoder model is faked (loading ms-marco-MiniLM would
download a model); the reranker's own text handling, pairing, sorting and
top_n logic run for real. See tests/test_integration_models.py for the
real-model variant.
"""

import pytest

import reranker
from reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Deterministic model double: returns a fixed score per pair."""

    def __init__(self, scores):
        self._scores = list(scores)
        self.pairs_seen = []

    def predict(self, pairs):
        self.pairs_seen.append(list(pairs))
        assert len(pairs) <= len(self._scores), "more pairs than scripted scores"
        return list(self._scores[: len(pairs)])


@pytest.fixture
def scripted_model(monkeypatch):
    """Replace the CrossEncoder model class; returns the fake instance."""

    def _install(scores):
        model = FakeCrossEncoder(scores)
        monkeypatch.setattr(reranker, "CrossEncoder", lambda name: model)
        return model

    return _install


class _Doc:
    """LangChain Document-like object."""

    def __init__(self, page_content):
        self.page_content = page_content


class TestRerankOrdering:
    def test_sorts_by_model_score_descending(self, scripted_model):
        scripted_model([0.1, 0.9, 0.5])
        docs = [_Doc("first"), _Doc("second"), _Doc("third")]
        out = CrossEncoderReranker().rerank("q", docs, top_n=3)
        assert [doc.page_content for _score, doc in out] == ["second", "third", "first"]
        scores = [score for score, _doc in out]
        assert scores == sorted(scores, reverse=True)

    def test_pairs_query_with_each_document_text(self, scripted_model):
        model = scripted_model([0.5, 0.4])
        docs = ["plain string", {"text": "from dict"}]
        CrossEncoderReranker().rerank("return policy", docs, top_n=2)
        assert model.pairs_seen == [
            [("return policy", "plain string"), ("return policy", "from dict")]
        ]

    def test_top_n_truncates_after_sorting(self, scripted_model):
        # Lowest score sits on the first document; it must be cut, not kept.
        scripted_model([0.0, 0.8, 0.9])
        docs = [_Doc("a"), _Doc("b"), _Doc("c")]
        out = CrossEncoderReranker().rerank("q", docs, top_n=2)
        assert [doc.page_content for _score, doc in out] == ["c", "b"]

    def test_default_top_n_is_three(self, scripted_model):
        scripted_model([0.1, 0.2, 0.3, 0.4, 0.5])
        docs = [_Doc(str(i)) for i in range(5)]
        assert len(CrossEncoderReranker().rerank("q", docs)) == 3


class TestRerankInputShapes:
    def test_accepts_plain_strings(self, scripted_model):
        scripted_model([1.0])
        out = CrossEncoderReranker().rerank("q", ["just text"], top_n=1)
        assert out == [(1.0, "just text")]

    def test_accepts_dict_variants(self, scripted_model):
        scripted_model([0.3, 0.4, 0.2, 0.1])
        docs = [
            {"text": "text key"},
            {"content": "content key"},
            {"page_content": "page_content key"},
            {"unrelated": "falls back to empty"},
        ]
        out = CrossEncoderReranker().rerank("q", docs, top_n=4)
        # Same fallback chain the reranker uses: text -> content -> page_content.
        # Order follows the scripted scores (0.4, 0.3, 0.2, 0.1).
        texts = [
            doc.get("text") or doc.get("content") or doc.get("page_content", "") for _s, doc in out
        ]
        assert texts == ["content key", "text key", "page_content key", ""]

    def test_accepts_document_like_objects(self, scripted_model):
        scripted_model([0.7])
        out = CrossEncoderReranker().rerank("q", [_Doc("doc text")], top_n=1)
        assert out[0][1].page_content == "doc text"


class TestRerankEdgeCases:
    def test_empty_document_list_returns_empty(self, scripted_model):
        scripted_model([])
        assert CrossEncoderReranker().rerank("q", [], top_n=3) == []

    def test_more_results_requested_than_documents(self, scripted_model):
        scripted_model([0.5])
        out = CrossEncoderReranker().rerank("q", [_Doc("only")], top_n=10)
        assert len(out) == 1
