"""Request observability: correlation ids, structured access logs, metrics."""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app_logging import set_request_id
from backend.monitoring import REQUEST_DURATION_SECONDS, REQUESTS_TOTAL

logger = logging.getLogger("marketplace.access")

METRICS_PATH = "/metrics"


def _endpoint_label(request: Request) -> str:
    """Route template for metrics/log labels, "unmatched" when no route matched.

    Raw paths are never used as labels: they embed review/listing ids and
    arbitrary scanner URLs, which would make metrics high-cardinality.
    """
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, emits one structured access log per request, and
    records request-count and latency metrics with bounded-cardinality labels."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path == METRICS_PATH:
            # Scrapes would otherwise pollute their own metrics and access logs.
            return await call_next(request)
        request_id = set_request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Unhandled exceptions are converted to a 500 response above this
            # middleware (Starlette's ServerErrorMiddleware), so anything that
            # reaches here is recorded, then re-raised.
            self._record(request, 500, start)
            raise
        self._record(request, response.status_code, start)
        response.headers["x-request-id"] = request_id
        return response

    @staticmethod
    def _record(request: Request, status: int, start: float) -> None:
        duration_s = time.perf_counter() - start
        endpoint = _endpoint_label(request)
        method = request.method
        REQUESTS_TOTAL.labels(method, endpoint, str(status)).inc()
        REQUEST_DURATION_SECONDS.labels(method, endpoint).observe(duration_s)
        logger.info(
            "http_request",
            extra={
                "extra_data": {
                    "method": method,
                    "path": request.url.path,
                    "status": status,
                    "duration_ms": int(duration_s * 1000),
                }
            },
        )
