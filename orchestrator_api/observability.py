import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from orchestrator_api.config import settings


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status_code", 
        "latency_ms", "client_ip", "user_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.setLevel(level)
    root.addHandler(handler)


@dataclass
class MetricsRegistry:
    requests_total: int = 0
    requests_inflight: int = 0
    requests_by_status: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rate_limited_total: int = 0
    request_latency_ms_sum: float = 0.0
    request_latency_ms_count: int = 0
    _lock: Lock = field(default_factory=Lock)

    def record_request(self, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_by_status[str(status_code)] += 1
            self.request_latency_ms_sum += latency_ms
            self.request_latency_ms_count += 1

    def set_inflight(self, delta: int) -> None:
        with self._lock:
            self.requests_inflight += delta
            if self.requests_inflight < 0:
                self.requests_inflight = 0

    def record_rate_limited(self) -> None:
        with self._lock:
            self.rate_limited_total += 1

    def render_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# TYPE http_requests_total counter",
                f"http_requests_total {self.requests_total}",
                "# TYPE http_requests_inflight gauge",
                f"http_requests_inflight {self.requests_inflight}",
                "# TYPE http_rate_limited_total counter",
                f"http_rate_limited_total {self.rate_limited_total}",
                "# TYPE http_request_latency_ms_sum counter",
                f"http_request_latency_ms_sum {self.request_latency_ms_sum}",
                "# TYPE http_request_latency_ms_count counter",
                f"http_request_latency_ms_count {self.request_latency_ms_count}",
                "# TYPE http_requests_by_status_total counter",
            ]
            for code, count in sorted(self.requests_by_status.items()):
                lines.append(f'http_requests_by_status_total{{status="{code}"}} {count}')
            return "\n".join(lines) + "\n"


metrics_registry = MetricsRegistry()
_access_logger = logging.getLogger("orchestrator.access")


class RequestMetricsAndLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        metrics_registry.set_inflight(1)

        response: Response | None = None
        exc: Exception | None = None
        try:
            response = await call_next(request)
            return response
        except Exception as err:  # noqa: BLE001
            exc = err
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            status_code = response.status_code if response else 500
            metrics_registry.set_inflight(-1)
            if settings.enable_metrics:
                metrics_registry.record_request(status_code=status_code, latency_ms=latency_ms)

            extra = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
                "client_ip": request.client.host if request.client else None,
                "user_id": request.headers.get("x-user-id"),
            }
            if exc is None:
                _access_logger.info("request_complete", extra=extra)
            else:
                _access_logger.exception("request_failed", extra=extra)

            if response is not None:
                response.headers["x-request-id"] = request_id


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._lock = Lock()
        self._buckets: dict[str, deque[float]] = {}

    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)
        if request.url.path in {"/health", "/metrics"}:
            return await call_next(request)

        key = request.headers.get("x-user-id")
        if not key:
            key = request.client.host if request.client else "anonymous"
        now = time.time()

        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            window_start = now - float(settings.rate_limit_window_seconds)
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= settings.rate_limit_requests:
                metrics_registry.record_rate_limited()
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded"},
                    headers={"Retry-After": str(settings.rate_limit_window_seconds)},
                )
            bucket.append(now)

        return await call_next(request)
