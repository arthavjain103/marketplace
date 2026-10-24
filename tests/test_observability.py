"""Focused tests: logging, exception handlers, metrics, and reliability paths.

No real Groq/Pinecone calls are made; external services are faked.
"""

import json
import logging
import sqlite3

import pytest
from fastapi import FastAPI, HTTPException
from prometheus_client import REGISTRY
from starlette.testclient import TestClient

from app_logging import JsonFormatter, set_request_id
from backend import app as app_module
from backend.errors import DATABASE_ERROR_DETAIL, INTERNAL_ERROR_DETAIL, register_exception_handlers
from backend.middleware import ObservabilityMiddleware
from backend.monitoring import register_metrics

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _sample(name, labels):
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


def _probe_app() -> FastAPI:
    """Isolated app with the real middleware + handlers but no DB/models."""
    app = FastAPI()
    app.add_middleware(ObservabilityMiddleware)
    register_exception_handlers(app)
    register_metrics(app)

    @app.get("/items/{item_id}")
    def items(item_id: int):
        return {"id": item_id}

    @app.get("/missing")
    def missing():
        raise HTTPException(status_code=404, detail="Review not found.")

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret-boom-detail")

    @app.get("/db-fail")
    def db_fail():
        raise sqlite3.OperationalError("secret-disk-error")

    return app


@pytest.fixture()
def probe():
    return TestClient(_probe_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# app_logging: JsonFormatter + request ids
# ---------------------------------------------------------------------------


def _record(msg="evt", level=logging.ERROR, extra=None, exc=None):
    record = logging.LogRecord("test.logger", level, __file__, 1, msg, (), None)
    if extra:
        record.extra_data = extra
    if exc is not None:
        record.exc_info = exc
    return record


def test_json_formatter_event_and_data():
    out = json.loads(JsonFormatter().format(_record(extra={"k": "v", "n": 3})))
    assert out["event"] == "evt"
    assert out["level"] == "ERROR"
    assert out["data"] == {"k": "v", "n": 3}


def test_json_formatter_includes_exception_and_request_id():
    import sys

    set_request_id("rid-exc")
    try:
        raise ValueError("boom")
    except ValueError:
        out = json.loads(JsonFormatter().format(_record(exc=sys.exc_info())))
    assert "ValueError: boom" in out["exc"]
    assert out["request_id"] == "rid-exc"


def test_json_formatter_request_id_from_contextvar():
    set_request_id("abc-123")
    out = json.loads(JsonFormatter().format(_record()))
    assert out["request_id"] == "abc-123"


def test_json_formatter_never_crashes_on_unserializable_data():
    out = json.loads(JsonFormatter().format(_record(extra={"obj": object()})))
    assert "data" in out


def test_set_request_id_rejects_unsafe_values():
    assert set_request_id("bad id with space").count(" ") == 0
    assert set_request_id("x" * 65) != "x" * 65
    assert len(set_request_id(None)) == 32


# ---------------------------------------------------------------------------
# Middleware: request id, access log, metrics
# ---------------------------------------------------------------------------


def test_request_id_echoed_when_safe(probe):
    resp = probe.get("/items/7", headers={"X-Request-ID": "req-test-123"})
    assert resp.headers["x-request-id"] == "req-test-123"


def test_request_id_minted_when_unsafe(probe):
    resp = probe.get("/items/7", headers={"X-Request-ID": "bad id with space"})
    minted = resp.headers["x-request-id"]
    assert minted != "bad id with space"
    assert len(minted) == 32


def test_access_log_carries_method_path_status_latency(probe, caplog):
    with caplog.at_level(logging.INFO, logger="marketplace.access"):
        probe.get("/items/7")
    rec = next(r for r in caplog.records if r.getMessage() == "http_request")
    assert rec.extra_data["method"] == "GET"
    assert rec.extra_data["path"] == "/items/7"
    assert rec.extra_data["status"] == 200
    assert rec.extra_data["duration_ms"] >= 0


def test_metrics_labels_use_route_template_not_raw_path(probe):
    labels = {"method": "GET", "endpoint": "/items/{item_id}", "status": "200"}
    before = _sample("http_requests_total", labels)
    probe.get("/items/42")
    probe.get("/items/43")
    assert _sample("http_requests_total", labels) - before == 2
    assert _sample("http_requests_total", {"method": "GET", "endpoint": "/items/42"}) == 0.0


def test_unmatched_routes_share_one_low_cardinality_label(probe):
    labels = {"method": "GET", "endpoint": "unmatched", "status": "404"}
    before = _sample("http_requests_total", labels)
    probe.get("/definitely-not-a-route-1")
    probe.get("/definitely-not-a-route-2")
    assert _sample("http_requests_total", labels) - before == 2


def test_latency_histogram_recorded_per_route_template(probe):
    labels = {"method": "GET", "endpoint": "/items/{item_id}"}
    before = _sample("http_request_duration_seconds_count", labels)
    probe.get("/items/1")
    assert _sample("http_request_duration_seconds_count", labels) - before == 1


def test_metrics_endpoint_exposed_and_self_excluded():
    client = TestClient(app_module.app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "http_requests_total" in resp.text
    assert _sample("http_requests_total", {"method": "GET", "endpoint": "/metrics"}) == 0.0


def test_exceptions_counter_records_endpoint_and_type(probe):
    labels = {"endpoint": "/boom", "exception": "RuntimeError"}
    before = _sample("app_exceptions_total", labels)
    probe.get("/boom")
    assert _sample("app_exceptions_total", labels) - before == 1


# ---------------------------------------------------------------------------
# Exception handlers: safe bodies, status codes, logs
# ---------------------------------------------------------------------------


def test_unhandled_exception_returns_safe_500(probe, caplog):
    with caplog.at_level(logging.ERROR, logger="marketplace.api.errors"):
        resp = probe.get("/boom", headers={"X-Request-ID": "rid-500"})
    body = resp.json()
    assert resp.status_code == 500
    assert body["detail"] == INTERNAL_ERROR_DETAIL
    assert body["request_id"] == "rid-500"
    assert "secret-boom-detail" not in resp.text
    assert "RuntimeError" not in resp.text
    rec = next(r for r in caplog.records if r.getMessage() == "unhandled_exception")
    assert rec.exc_info is not None  # stack trace stays in server logs
    assert rec.extra_data["path"] == "/boom"


def test_http_exception_keeps_detail_shape_and_logs(probe, caplog):
    with caplog.at_level(logging.INFO, logger="marketplace.api.errors"):
        resp = probe.get("/missing")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Review not found."}
    rec = next(r for r in caplog.records if r.getMessage() == "http_error")
    assert rec.extra_data["status"] == 404
    assert rec.extra_data["detail"] == "Review not found."


def test_validation_errors_keep_default_422_shape():
    client = TestClient(app_module.app)
    resp = client.post("/api/ask", json={"query": "", "top_n": 6})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, list) and detail


def test_database_errors_map_to_503_without_leaking_message(probe, caplog):
    with caplog.at_level(logging.ERROR, logger="marketplace.api.errors"):
        resp = probe.get("/db-fail")
    assert resp.status_code == 503
    assert resp.json() == {"detail": DATABASE_ERROR_DETAIL}
    assert "secret-disk-error" not in resp.text
    rec = next(r for r in caplog.records if r.getMessage() == "database_error")
    assert rec.exc_info is not None


def test_database_failures_via_store_map_to_503(monkeypatch):
    import store

    def broken_get_db(path=store.DEFAULT_DB):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "get_db", broken_get_db)
    monkeypatch.setattr(app_module, "_conn", None)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    resp = client.get("/api/listings")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Endpoint reliability: /api/init and /api/ask
# ---------------------------------------------------------------------------


def test_init_missing_artifact_logs_error_and_keeps_500(monkeypatch, caplog):
    monkeypatch.setattr(app_module.os.path, "exists", lambda path: False)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="marketplace.api"):
        resp = client.post("/api/init")
    assert resp.status_code == 500
    assert app_module._state["loaded"] is False
    assert any(r.getMessage() == "models_init_failed" for r in caplog.records)


def test_init_failure_logs_stack_and_returns_503(monkeypatch, caplog):
    monkeypatch.setattr(app_module.os.path, "exists", lambda path: True)

    def broken_load(fileobj):
        raise ValueError("corrupt pickle")

    monkeypatch.setattr(app_module.pickle, "load", broken_load)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="marketplace.api"):
        resp = client.post("/api/init")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Model initialization failed. Check server logs."
    rec = next(r for r in caplog.records if r.getMessage() == "models_init_failed")
    assert rec.exc_info is not None
    assert app_module._state["loaded"] is False


class _BrokenSearch:
    def search(self, query, top_n=6):
        raise ConnectionError("vector db down")


class _EmptySearch:
    def search(self, query, top_n=6):
        return []


def test_ask_returns_503_when_retrieval_fails(monkeypatch, caplog):
    monkeypatch.setitem(app_module._state, "loaded", True)
    monkeypatch.setitem(app_module._state, "hybrid", _BrokenSearch())
    client = TestClient(app_module.app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="marketplace.api"):
        resp = client.post("/api/ask", json={"query": "refund policy", "top_n": 3})
    assert resp.status_code == 503
    assert "vector db down" not in resp.text
    rec = next(r for r in caplog.records if r.getMessage() == "ask_retrieval_failed")
    assert rec.exc_info is not None


def test_ask_returns_502_when_llm_fails(monkeypatch, caplog):
    monkeypatch.setitem(app_module._state, "loaded", True)
    monkeypatch.setitem(app_module._state, "hybrid", _EmptySearch())

    def broken_answer(question, context):
        raise RuntimeError("groq down")

    monkeypatch.setattr(app_module, "generate_answer", broken_answer)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    with caplog.at_level(logging.WARNING, logger="marketplace.api"):
        resp = client.post("/api/ask", json={"query": "refund policy", "top_n": 3})
    assert resp.status_code == 502
    assert "groq down" not in resp.text
    assert any(r.getMessage() == "ask_answer_unavailable" for r in caplog.records)


# ---------------------------------------------------------------------------
# Sentry configuration
# ---------------------------------------------------------------------------


def test_sentry_disabled_without_dsn(monkeypatch, caplog):
    import backend.monitoring as monitoring

    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.setattr(monitoring, "_sentry_client", None)
    with caplog.at_level(logging.INFO, logger="marketplace.monitoring"):
        assert monitoring.init_sentry() is None
    assert monitoring._sentry_client is None
    assert any(r.getMessage() == "sentry_disabled" for r in caplog.records)


def test_sentry_initialized_when_dsn_set(monkeypatch):
    import sentry_sdk

    import backend.monitoring as monitoring

    kwargs_captured = {}

    def fake_init(**kwargs):
        kwargs_captured.update(kwargs)

    monkeypatch.setattr(sentry_sdk, "init", fake_init)
    monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "test-env")
    assert monitoring.init_sentry() is sentry_sdk
    assert kwargs_captured["dsn"] == "https://key@example.ingest.sentry.io/1"
    assert kwargs_captured["environment"] == "test-env"
    assert kwargs_captured["send_default_pii"] is False
    assert len(kwargs_captured["integrations"]) == 3
    assert monitoring._sentry_client is sentry_sdk
    monitoring._sentry_client = None


def test_capture_exception_noop_without_sentry():
    import backend.monitoring as monitoring

    assert monitoring.capture_exception(ValueError("x")) is None


def test_capture_exception_never_raises(monkeypatch, caplog):
    import backend.monitoring as monitoring

    class FakeSentry:
        def capture_exception(self, error):
            raise RuntimeError("sentry transport down")

    monkeypatch.setattr(monitoring, "_sentry_client", FakeSentry())
    with caplog.at_level(logging.ERROR, logger="marketplace.monitoring"):
        monitoring.capture_exception(ValueError("app error"))
    assert any(r.getMessage() == "sentry_capture_failed" for r in caplog.records)


# ---------------------------------------------------------------------------
# store: database failure logging
# ---------------------------------------------------------------------------


def test_store_logs_db_errors_and_reraises(tmp_path, caplog):
    import store

    conn = store.get_db(str(tmp_path / "t.db"))
    conn.close()
    with caplog.at_level(logging.ERROR, logger="marketplace.store"):
        with pytest.raises(sqlite3.ProgrammingError):
            store.save_listing(conn, {"title": "t"})
    rec = next(r for r in caplog.records if r.getMessage() == "db_error")
    assert rec.exc_info is not None
    assert rec.extra_data["op"] == "save_listing"
