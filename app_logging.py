"""Structured JSON logging for app events and AI workflow steps."""

import contextvars
import json
import logging
import re
import sys
import uuid
from datetime import UTC, datetime

# Correlation id for the request currently being served; set per request by
# backend.middleware.ObservabilityMiddleware and read by JsonFormatter.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Client-supplied request ids are echoed only if they are plain header-safe
# tokens, so arbitrary strings (control chars, log-formatting tricks) cannot
# reach the logs.
_SAFE_REQUEST_ID = re.compile(r"^[0-9a-zA-Z_.-]{1,64}$")


def get_request_id():
    return request_id_var.get()


def set_request_id(value):
    """Adopt a client-supplied request id when safe; otherwise mint one."""
    if value and _SAFE_REQUEST_ID.fullmatch(value):
        request_id_var.set(value)
        return value
    fresh = uuid.uuid4().hex
    request_id_var.set(fresh)
    return fresh


class JsonFormatter(logging.Formatter):
    """One JSON object per record: event, optional data dict, optional exc."""

    def format(self, record):
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
        }
        data = getattr(record, "extra_data", None)
        if data:
            payload["data"] = data
        request_id = request_id_var.get()
        if request_id and "request_id" not in payload.get("data", {}):
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup(level="INFO"):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return root
