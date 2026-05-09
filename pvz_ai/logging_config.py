import logging
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from pvz_ai.config import Settings

SENSITIVE_WORDS = ("api_key", "apikey", "authorization", "bearer", "token", "secret")


def redact(value: str) -> str:
    lowered = value.lower()
    if any(word in lowered for word in SENSITIVE_WORDS):
        return "[redacted]"
    return value


def setup_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level.upper(),
        format=(
            "%(asctime)s %(levelname)s "
            "request_id=%(request_id)s session_id=%(session_id)s %(name)s %(message)s"
        ),
    )
    context_filter = MissingContextFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(context_filter)


class MissingContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "session_id"):
            record.session_id = "-"
        return True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid4())
        start = time.perf_counter()
        logger = logging.getLogger("pvz_ai.request")

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "request failed method=%s path=%s elapsed_ms=%s",
                request.method,
                request.url.path,
                elapsed_ms,
                extra={"request_id": request_id},
            )
            raise

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request completed method=%s path=%s status=%s elapsed_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            extra={"request_id": request_id},
        )
        return response
