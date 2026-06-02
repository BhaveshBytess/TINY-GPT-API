"""
Global exception handlers.

Catches specific exception types raised anywhere in the app and converts
them to clean HTTP responses. Endpoints can raise these exceptions without
needing try/except — the handler does the conversion.
"""
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.cloud_client import CloudAPIError


logger = logging.getLogger(__name__)


async def cloud_api_error_handler(request: Request, exc: CloudAPIError) -> JSONResponse:
    """Handle cloud API failures uniformly across the app."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "Cloud API error: %s [request_id=%s, path=%s]",
        str(exc), request_id, request.url.path
    )
    
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Cloud API unavailable",
            "error": str(exc),
            "request_id": request_id,
        },
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle programmer-thrown ValueErrors as 400 Bad Request."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        "ValueError: %s [request_id=%s, path=%s]",
        str(exc), request_id, request.url.path
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Invalid input",
            "error": str(exc),
            "request_id": request_id,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions. Don't leak internals to users."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log with full traceback (logger.exception adds it automatically)
    logger.exception(
        "Unhandled exception [request_id=%s, path=%s]",
        request_id, request.url.path
    )
    
    # Don't include str(exc) — it might leak file paths, queries, etc.
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the app."""
    app.add_exception_handler(CloudAPIError, cloud_api_error_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)