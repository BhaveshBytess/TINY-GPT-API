"""
Custom middleware that runs for every request.

Middleware in FastAPI is code that runs BEFORE and AFTER every endpoint.
It's the same concept as Express middleware in Node, or Django middleware.
Use cases:
  - Authentication (check tokens before route handler)
  - Logging (log request before, response after)
  - CORS (add headers to responses)
  - Request IDs (assign unique ID, attach to request and logs)
"""
import time
import uuid
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Adds a unique request ID to every request and logs the lifecycle.
    
    Lifecycle:
      1. Generate a short unique ID (8 chars, e.g., 'a3f1c2d8')
      2. Attach it to the request object so endpoints can read it
      3. Log "Request started" with method, path, ID
      4. Call the actual endpoint (await call_next)
      5. Log "Request completed" with status code, duration, ID
      6. Add the ID to the response headers (so clients can reference it)
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate request ID (first 8 chars of a UUID)
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        
        # Log start
        start_time = time.time()
        logger.info(
            "Request started: %s %s [request_id=%s]",
            request.method, request.url.path, request_id
        )
        
        # Call the endpoint
        try:
            response = await call_next(request)
        except Exception as exc:
            # Anything not caught by endpoint handlers comes here
            duration_ms = int((time.time() - start_time) * 1000)
            logger.exception(
                "Request failed with unhandled exception: %s %s "
                "[request_id=%s, duration=%dms]",
                request.method, request.url.path, request_id, duration_ms
            )
            raise  # Re-raise so FastAPI's default handler responds
        
        # Log completion
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Request completed: %s %s -> %d [request_id=%s, duration=%dms]",
            request.method, request.url.path, response.status_code,
            request_id, duration_ms
        )
        
        # Add request ID to response headers (so the client can reference it)
        response.headers["X-Request-ID"] = request_id
        
        return response