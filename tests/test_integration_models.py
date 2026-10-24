"""Optional integration tests against the real downloadable models/services.

These are marked ``integration`` and skipped unless ``RUN_MODEL_TESTS=1``,
so the default suite (and CI) never downloads models or needs API keys.

Run them locally with::

    RUN_MODEL_TESTS=1 uv run pytest tests/ -m integration -v

First run downloads intfloat/multilingual-e5-base (~1GB) and
cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB) into the local HuggingFace
cache. The Pinecone test additionally needs PINECONE_API_KEY and
PINECONE_INDEX (from .env) and skips itself when they are absent.
"""

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_TESTS") != "1",
        reason="downloads real models; set RUN_MODEL_TESTS=1 to enable",
    ),
]


class TestRealE5Embeddings:
    def test_embedding_dimension_is_768(self):
        from ingestion.embedder import get_embedding_model

        model = get_embedding_model()
        vec = model.embed_query("price must be a positive number")
        assert len(vec) == 768

    def test_query_vectors_are_normalized(self):
        import math

        from ingestion.embedder import get_embedding_model

        model = get_embedding_model()
        vec = model.embed_query("shipping policy")
        assert abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-6

    def test_batch_shape_matches_input(self):
        from ingestion.embedder import get_embedding_model

        model = get_embedding_model()
        out = model.embed_documents(["first chunk", "second chunk"])
        assert len(out) == 2
        assert all(len(vec) == 768 for vec in out)


class TestRealCrossEncoderReranker:
    def test_relevant_document_ranks_first(self):
        from reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        docs = [
            "Tomato pasta recipe with fresh basil and parmesan.",
            "If the item arrives damaged you may return it for a full refund.",
        ]
        query = "how do I return a damaged item for a refund?"
        (score, text), *_ = reranker.rerank(query, docs, top_n=2)
        assert "refund" in text


class TestRealSemanticSearch:
    def test_search_against_pinecone_index(self):
        if not (os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX")):
            pytest.skip("PINECONE_API_KEY / PINECONE_INDEX not configured")
        from pro.semantic import SemanticSearch

        search = SemanticSearch()
        results = search.search("price rules for electronics", top_n=2)
        assert isinstance(results, list)
        assert len(results) <= 2
        for doc, _score in results:
            assert hasattr(doc, "page_content")
