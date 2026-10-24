"""Integration tests for the retrieval pipeline.

These exercise the project's real implementations end to end:

    chunker.split_policy_document -> KeywordSearchBM25 (real BM25Okapi)
    -> CombinedSearch RRF fusion -> CrossEncoderReranker (real class)

Only the true external boundaries are faked:
  * the Pinecone-backed vector search (network service) is a stub honoring
    the same ``search(query, top_n) -> [(Document, score), ...]`` contract;
  * the neural model inside CrossEncoderReranker is replaced with a
    deterministic query-term-overlap model (loading ms-marco-MiniLM would
    download a model — the real one runs in tests/test_integration_models.py).

The existing unit tests in test_hybrid.py re-implement the fusion math to
inspect it; these tests call ``CombinedSearch.search`` for real instead, so
they would catch regressions in the fusion code itself.
"""

import pytest
from langchain_core.documents import Document

import reranker as reranker_module
from ingestion.chunker import chunk_documents, is_policy_doc
from pro.hybrid import CombinedSearch
from pro.keyword_based import KeywordSearchBM25

pytestmark = pytest.mark.integration

POLICY_TEXT = """\
Part A. Policy
1. General

POL-1.1 Listings must include the item condition.
POL-1.2 Titles must be written in English.

2. Pricing

POL-2.1 Price must be a positive number.
POL-2.2 Price must not exceed 100000.
POL-2.3 Price must use at most two decimal places.

3. Shipping

POL-3.1 Sellers must ship within three business days.
POL-3.2 A tracking number is required for electronics.

Part B. Brand Guide
4. Voice

BRD-1.1 Keep the tone friendly and factual.
BRD-1.2 Avoid exclamation marks in titles.

5. Imagery

BRD-2.1 Photos must show the actual product.
BRD-2.2 Stock photos are not allowed for handmade items.
"""


class _OverlapModel:
    """Deterministic cross-encoder double: score = query words found in text."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            words = set(str(query).lower().split())
            scores.append(float(sum(1 for w in words if w in str(text).lower())))
        return scores


class _FakeVectorSearch:
    """Stands in for pro.semantic.SemanticSearch (Pinecone is external)."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, query, top_n=5):
        self.calls.append({"query": query, "top_n": top_n})
        return self._results[:top_n]


def _policy_chunks():
    doc = Document(page_content=POLICY_TEXT, metadata={"source": "tests/fixtures/policy.pdf"})
    return chunk_documents([doc])


@pytest.fixture
def keyword_search():
    chunks = _policy_chunks()
    return KeywordSearchBM25(
        documents=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )


@pytest.fixture
def hybrid(keyword_search, monkeypatch):
    """CombinedSearch with real BM25 + fake Pinecone + real reranker code."""
    monkeypatch.setattr(reranker_module, "CrossEncoder", lambda name: _OverlapModel())
    vector = _FakeVectorSearch([])
    return CombinedSearch(keyword_search, vector)


def _rule_chunk(rule_id):
    return next(c for c in _policy_chunks() if c.metadata.get("rule_id") == rule_id)


# ---------------------------------------------------------------------------
# Chunking -> BM25 index
# ---------------------------------------------------------------------------


class TestPolicyCorpus:
    def test_corpus_is_detected_and_structurally_chunked(self):
        assert is_policy_doc(POLICY_TEXT) is True
        rules = [c for c in _policy_chunks() if c.metadata["chunk_type"] == "rule"]
        assert len(rules) == 11  # 7 POL + 4 BRD, one chunk per rule
        assert len({c.metadata["chunk_id"] for c in rules}) == 11

    def test_doc_type_follows_rule_prefix(self):
        chunks = {
            c.metadata["rule_id"]: c for c in _policy_chunks() if c.metadata["chunk_type"] == "rule"
        }
        assert chunks["POL-2.1"].metadata["doc_type"] == "policy"
        assert chunks["BRD-1.1"].metadata["doc_type"] == "brand_guide"


class TestKeywordRetrieval:
    def test_relevant_rule_ranks_first(self, keyword_search):
        results = keyword_search.search("price must be a positive number", top_n=3)
        assert results[0][2]["rule_id"] == "POL-2.1"
        assert results[0][0] > 0

    def test_metadata_travels_with_results(self, keyword_search):
        top = keyword_search.search("price must be a positive number", top_n=1)[0]
        assert top[2]["source"] == "policy.pdf"
        assert top[2]["chunk_type"] == "rule"

    def test_unmatched_query_returns_zero_scores(self, keyword_search):
        results = keyword_search.search("zzz qqq", top_n=2)
        assert len(results) == 2
        assert all(score == 0.0 for score, _text, _meta in results)

    def test_top_n_larger_than_corpus_returns_everything(self, keyword_search):
        assert len(keyword_search.search("price", top_n=100)) == 11

    def test_top_n_zero_returns_nothing(self, keyword_search):
        assert keyword_search.search("price", top_n=0) == []


# ---------------------------------------------------------------------------
# CombinedSearch end to end (fusion + dedup + reranking)
# ---------------------------------------------------------------------------


class TestHybridSearchIntegration:
    def test_duplicates_across_sources_merge_into_one_result(self, hybrid):
        pol21 = _rule_chunk("POL-2.1")
        # The same chunk text arrives from both BM25 and the vector store.
        vector = _FakeVectorSearch(
            [(Document(page_content=pol21.page_content, metadata={"via": "pinecone"}), 0.9)]
        )
        hybrid.vector_search = vector
        results = hybrid.search("price must be a positive number", top_n=5)
        texts = [item["text"] for _score, item in results]
        assert texts.count(pol21.page_content) == 1

    def test_vector_only_results_are_included(self, hybrid):
        brd = _rule_chunk("BRD-1.1")
        hybrid.vector_search = _FakeVectorSearch(
            [(Document(page_content=brd.page_content, metadata=brd.metadata), 0.9)]
        )
        results = hybrid.search("tone of voice rules", top_n=5)
        assert any(item["metadata"].get("rule_id") == "BRD-1.1" for _score, item in results)

    def test_bm25_metadata_survives_to_output(self, hybrid):
        # Vector result carries no metadata, so the BM25 metadata is kept.
        hybrid.vector_search = _FakeVectorSearch(
            [(Document(page_content="unrelated vector text"), 0.9)]
        )
        results = hybrid.search("price must be a positive number", top_n=3)
        top = results[0][1]
        assert top["text"].startswith("[2. Pricing | POL-2.1]")
        assert top["metadata"]["rule_id"] == "POL-2.1"

    def test_results_are_sorted_by_reranker_score(self, hybrid):
        results = hybrid.search("price must be a positive number", top_n=5)
        scores = [score for score, _item in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_is_respected(self, hybrid):
        assert len(hybrid.search("price tracking shipping", top_n=2)) <= 2
        assert hybrid.search("price tracking shipping", top_n=0) == []

    def test_unmatched_query_still_returns_bm25_corpus_order(self, hybrid):
        # Real BM25 returns zero-score docs for unknown tokens; CombinedSearch
        # does not drop them — callers see the corpus in fused order.
        results = hybrid.search("zzz qqq", top_n=3)
        assert len(results) == 3
        assert all(score == 0.0 for score, _item in results)

    def test_vector_failure_propagates_to_caller(self, hybrid):
        class Broken:
            def search(self, query, top_n=5):
                raise ConnectionError("pinecone down")

        hybrid.vector_search = Broken()
        with pytest.raises(ConnectionError):
            hybrid.search("price", top_n=5)

    def test_fetch_k_covers_top_n_expansion(self, hybrid):
        hybrid.search("price", top_n=3)
        assert hybrid.vector_search.calls[0]["top_n"] == 15  # top_n * 5
