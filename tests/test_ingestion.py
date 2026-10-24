"""Tests for the ingestion pipeline pieces: embedder, chunker routing,
pdf_loader, pinecone_upload.

External boundaries are faked (HuggingFace model weights, Pinecone client);
everything else runs the project's real implementation.
"""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfWriter

import config
from ingestion.chunker import chunk_documents, is_policy_doc
from ingestion.embedder import E5Embeddings
from ingestion.pdf_loader import load_documents
from ingestion.pinecone_upload import _clean_metadata, upload

# ---------------------------------------------------------------------------
# E5Embeddings: prefix contract (model weights faked via the parent class)
# ---------------------------------------------------------------------------


def _bare_e5():
    """E5Embeddings without __init__, so no model download happens."""
    return object.__new__(E5Embeddings)


def test_embed_documents_adds_passage_prefix(monkeypatch):
    captured = {}

    def fake_embed_documents(self, texts):
        captured["texts"] = list(texts)
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(HuggingFaceEmbeddings, "embed_documents", fake_embed_documents)
    out = _bare_e5().embed_documents(["first doc", "second doc"])
    assert captured["texts"] == ["passage: first doc", "passage: second doc"]
    assert out == [[0.1, 0.2], [0.1, 0.2]]


def test_embed_query_adds_query_prefix(monkeypatch):
    captured = {}

    def fake_embed_query(self, text):
        captured["text"] = text
        return [0.3, 0.4]

    monkeypatch.setattr(HuggingFaceEmbeddings, "embed_query", fake_embed_query)
    assert _bare_e5().embed_query("refund policy") == [0.3, 0.4]
    assert captured["text"] == "query: refund policy"


def test_embed_output_shape_is_preserved(monkeypatch):
    def fake_embed_documents(self, texts):
        return [[1.0] * 8 for _ in texts]

    monkeypatch.setattr(HuggingFaceEmbeddings, "embed_documents", fake_embed_documents)
    out = _bare_e5().embed_documents(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(vec) == 8 for vec in out)


def test_embed_empty_input_is_handled(monkeypatch):
    captured = {}

    def fake_embed_documents(self, texts):
        captured["texts"] = list(texts)
        return []

    monkeypatch.setattr(HuggingFaceEmbeddings, "embed_documents", fake_embed_documents)
    assert _bare_e5().embed_documents([]) == []
    assert captured["texts"] == []


# ---------------------------------------------------------------------------
# chunk_documents routing: policy / email / fallback splitter
# ---------------------------------------------------------------------------

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

EMAIL_TEXT = """Template 1: Order Shipped
🔍 auto-message
Your order has shipped. Expect delivery in 3-5 business days.
"""


class TestChunkDocumentsRouting:
    def test_policy_document_uses_structural_chunking(self):
        doc = Document(page_content=POLICY_TEXT, metadata={"source": "policy.pdf"})
        chunks = chunk_documents([doc])
        types = {c.metadata["chunk_type"] for c in chunks}
        assert "rule" in types
        # Every chunk keeps the source basename, not the full path.
        assert all(c.metadata["source"] == "policy.pdf" for c in chunks)

    def test_email_document_uses_template_chunking(self):
        doc = Document(page_content=EMAIL_TEXT, metadata={"source": "emails.pdf", "skip": None})
        chunks = chunk_documents([doc])
        types = [c.metadata["chunk_type"] for c in chunks]
        assert "template" in types and "embedded_annotation" in types
        template = next(c for c in chunks if c.metadata["chunk_type"] == "template")
        assert template.metadata["template_num"] == "1"
        assert template.metadata["template_name"] == "Order Shipped"
        # Metadata values of None are dropped (Pinecone disallows them).
        assert "skip" not in template.metadata

    def test_plain_document_uses_fallback_splitter(self):
        text = " ".join(f"Sentence number {i} about ordinary things." for i in range(60))
        doc = Document(page_content=text, metadata={"source": "notes.pdf"})
        chunks = chunk_documents([doc])
        assert len(chunks) >= 2
        assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)

    def test_empty_document_list_returns_empty(self):
        assert chunk_documents([]) == []

    def test_policy_detection_threshold(self):
        assert is_policy_doc(POLICY_TEXT) is True
        nine_rules = "\n".join(f"POL-{i}.1 Some rule." for i in range(1, 10))
        assert is_policy_doc(nine_rules) is False


class TestRuleChunkMetadata:
    def test_rule_ids_are_unique_and_typed(self):
        doc = Document(page_content=POLICY_TEXT, metadata={"source": "policy.pdf"})
        chunks = chunk_documents([doc])
        rules = [c for c in chunks if c.metadata["chunk_type"] == "rule"]
        rule_ids = [c.metadata["rule_id"] for c in rules]
        assert len(rule_ids) == len(set(rule_ids))
        assert "POL-2.1" in rule_ids and "BRD-1.1" in rule_ids
        by_id = {c.metadata["rule_id"]: c for c in rules}
        assert by_id["POL-2.1"].metadata["doc_type"] == "policy"
        assert by_id["BRD-1.1"].metadata["doc_type"] == "brand_guide"

    def test_each_rule_text_carries_header_and_rule_id(self):
        doc = Document(page_content=POLICY_TEXT, metadata={"source": "policy.pdf"})
        chunks = chunk_documents([doc])
        pol21 = next(c for c in chunks if c.metadata.get("rule_id") == "POL-2.1")
        assert pol21.page_content.startswith("[2. Pricing | POL-2.1]")
        assert "POL-2.1: Price must be a positive number." in pol21.page_content


# ---------------------------------------------------------------------------
# pdf_loader.load_documents: real PDFs built with pypdf (no downloads)
# ---------------------------------------------------------------------------


def _write_pdf(path, pages):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        writer.write(f)


class TestLoadDocuments:
    def test_merges_pages_of_one_pdf(self, tmp_path):
        _write_pdf(tmp_path / "two_pages.pdf", 2)
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata["num_pages"] == 2
        assert "page" not in docs[0].metadata

    def test_keeps_one_document_per_source_file(self, tmp_path):
        _write_pdf(tmp_path / "a.pdf", 2)
        _write_pdf(tmp_path / "b.pdf", 1)
        docs = load_documents(str(tmp_path))
        assert len(docs) == 2
        num_pages = sorted(d.metadata["num_pages"] for d in docs)
        assert num_pages == [1, 2]


# ---------------------------------------------------------------------------
# pinecone_upload: metadata cleaning + stable id scheme (Pinecone faked)
# ---------------------------------------------------------------------------


def test_clean_metadata_keeps_pinecone_compatible_values_only():
    meta = {
        "text": "keep",
        "none": None,
        "empty_str": "",
        "empty_list": [],
        "empty_tuple": (),
        "number": 3,
        "flag": True,
        "list_mixed": ["x", 2],
        "dict": {"k": 1},
        "a_set": {"b"},
    }
    out = _clean_metadata(meta)
    assert out == {
        "text": "keep",
        "number": 3,
        "flag": True,
        "list_mixed": ["x", "2"],
        "dict": str({"k": 1}),
        "a_set": ["b"],
    }


def test_upload_assigns_stable_ids_and_cleans_metadata(monkeypatch):
    import ingestion.pinecone_upload as pc

    captured = {}

    def fake_from_documents(chunks, embedding, index_name, ids, batch_size, namespace):
        captured.update(
            chunks=chunks, ids=ids, index=index_name, namespace=namespace, batch_size=batch_size
        )

    monkeypatch.setattr(pc.PineconeVectorStore, "from_documents", staticmethod(fake_from_documents))
    monkeypatch.setattr(pc, "get_embedding_model", lambda: "fake-embedder")

    chunks = [
        Document(
            page_content="rule text",
            metadata={"source": "sub\\dir\\policy.pdf", "chunk_id": "POL-1.1", "junk": None},
        ),
        Document(page_content="no chunk id", metadata={"source": "sub\\dir\\policy.pdf"}),
    ]
    upload(chunks, batch_size=7)

    # Ids are stable: re-ingesting overwrites instead of duplicating.
    # The id uses the source basename; metadata keeps the original source.
    assert captured["ids"] == ["policy.pdf::POL-1.1", "policy.pdf::1"]
    assert captured["chunks"][0].metadata == {
        "source": "sub\\dir\\policy.pdf",
        "chunk_id": "POL-1.1",
    }
    assert captured["index"] == config.PINECONE_INDEX
    assert captured["namespace"] is None
    assert captured["batch_size"] == 7
