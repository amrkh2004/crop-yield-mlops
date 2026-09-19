import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from prodml.logging import get_logger

logger = get_logger("prodml.api")


class LoggingAndCorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware that attaches a correlation ID (request_id) to each request,
    calculates execution latency in milliseconds, and emits structured logs.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        # Retrieve correlation ID from header or generate a new UUID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)

        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Include X-Request-ID in response header for client correlation
        response.headers["X-Request-ID"] = request_id

        # Log request details in structured JSON format
        logger.info(
            "request_processed",
            request_id=request_id,
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=process_time_ms,
        )

        return response
