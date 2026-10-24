"""Monitoring setup: Prometheus metrics plus optional Sentry error tracking.

Both are driven by environment variables so local development needs no
monitoring services: Sentry only initializes when SENTRY_DSN is set, and the
metrics registry is always available but cheap.
"""

import logging
import os

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

logger = logging.getLogger("marketplace.monitoring")

# Labels are route templates ("/api/reviews/{review_id}") and status codes
# only — never user ids, raw paths or listing content, so cardinality is
# bounded by the number of routes.
REQUESTS_TOTAL = Counter(
    "http_requests",
    "Total HTTP requests processed.",
    ["method", "endpoint", "status"],
)
REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "End-to-end request latency in seconds.",
    ["method", "endpoint"],
)
APP_EXCEPTIONS = Counter(
    "app_exceptions",
    "Unhandled exceptions converted to HTTP 500 responses.",
    ["endpoint", "exception"],
)


def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def register_metrics(app: FastAPI) -> None:
    """Add the /metrics route directly (include_router would nest a router)."""
    app.get("/metrics", include_in_schema=False)(metrics)


# Sentry is optional: capture_exception below is a no-op until init_sentry
# succeeds, so no code path needs to check whether Sentry is configured.
_sentry_client = None


def init_sentry():
    """Initialize Sentry when SENTRY_DSN is configured; otherwise stay off."""
    global _sentry_client
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        logger.info("sentry_disabled")
        return None
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError:  # pragma: no cover - only without the declared dependency
        logger.warning("sentry_sdk_not_installed")
        return None
    environment = os.getenv("SENTRY_ENVIRONMENT", "production")
    try:
        traces_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0"))
    except ValueError:
        traces_rate = 0.0
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_rate,
        send_default_pii=False,
        # Our handlers cover every exception: unhandled ones re-raise past
        # ServerErrorMiddleware and are captured by the ASGI layer, handled
        # ones (sqlite3.Error) are captured explicitly. Disabling automatic
        # failed-request events prevents duplicates and 503 noise.
        integrations=[
            StarletteIntegration(failed_request_status_codes=set()),
            FastApiIntegration(failed_request_status_codes=set()),
            # Error logs already land in our structured logs; only explicit
            # capture_exception calls become Sentry events.
            LoggingIntegration(level=logging.INFO, event_level=logging.CRITICAL),
        ],
    )
    _sentry_client = sentry_sdk
    logger.info("sentry_initialized", extra={"extra_data": {"environment": environment}})
    return sentry_sdk


def capture_exception(error):
    """Forward a handled exception to Sentry when enabled; never raises."""
    if _sentry_client is None:
        return
    try:
        _sentry_client.capture_exception(error)
    except Exception:
        logger.exception("sentry_capture_failed")
