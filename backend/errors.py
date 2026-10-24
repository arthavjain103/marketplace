"""Centralized FastAPI exception handling.

Client-visible responses never include stack traces, driver messages or
environment details: 5xx bodies are fixed phrases, context goes to the logs
(and Sentry) instead. 4xx responses keep FastAPI's {"detail": ...} shape so
existing clients are unaffected.
"""

import logging
import sqlite3

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app_logging import get_request_id
from backend.monitoring import APP_EXCEPTIONS, capture_exception

logger = logging.getLogger("marketplace.api.errors")

INTERNAL_ERROR_DETAIL = "Internal server error."
DATABASE_ERROR_DETAIL = "Storage is temporarily unavailable. Please try again later."


def _request_id(request: Request):
    return getattr(request.state, "request_id", None) or get_request_id()


def register_exception_handlers(app) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        level = logging.INFO if exc.status_code < 500 else logging.ERROR
        logger.log(
            level,
            "http_error",
            extra={
                "extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": exc.status_code,
                    "detail": str(exc.detail),
                }
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.info(
            "validation_error",
            extra={
                "extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": 422,
                    "errors": len(exc.errors()),
                }
            },
        )
        # Same shape as FastAPI's default handler; the payload is the caller's
        # own request data echoed back, not server internals.
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(sqlite3.Error)
    async def database_exception_handler(request: Request, exc: sqlite3.Error):
        logger.error(
            "database_error",
            exc_info=exc,
            extra={
                "extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": 503,
                    "exception": type(exc).__name__,
                }
            },
        )
        capture_exception(exc)
        return JSONResponse(status_code=503, content={"detail": DATABASE_ERROR_DETAIL})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        endpoint = getattr(request.scope.get("route"), "path", None) or "unmatched"
        APP_EXCEPTIONS.labels(endpoint, type(exc).__name__).inc()
        logger.error(
            "unhandled_exception",
            exc_info=exc,
            extra={
                "extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": 500,
                    "exception": type(exc).__name__,
                }
            },
        )
        # Unhandled exceptions re-raise past ServerErrorMiddleware and are
        # captured once by Sentry's ASGI layer; no explicit capture here.
        body = {"detail": INTERNAL_ERROR_DETAIL}
        request_id = _request_id(request)
        if request_id:
            body["request_id"] = request_id
        return JSONResponse(status_code=500, content=body)
