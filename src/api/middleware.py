"""API middleware: error handling, logging, CORS."""

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response: Response = await call_next(request)
        elapsed = time.time() - start
        logger.info(
            "%s %s -> %d (%.3fs)",
            request.method, request.url.path, response.status_code, elapsed
        )
        return response
