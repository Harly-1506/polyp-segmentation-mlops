from contextlib import contextmanager
from time import time
from typing import Iterator

from loguru import logger
try: 
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OTEL_AVAILABLE = True
except ImportError:  
    trace = None
    OTEL_AVAILABLE = False
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from .config import Settings

# Metrics definitions
REQUEST_COUNTER = Counter(
    "requests_total",
    "Total number of prediction requests received.",
    ["endpoint", "status"],
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Wall clock latency for inference round trips.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
HEALTH_GAUGE = Gauge(
    "service_health",
    "Represents service healthiness (1=healthy,0=unhealthy).",
    ["check"],
)

# basic code for monitoring and observability utilities
@contextmanager
def track_latency() -> Iterator[None]:
    """Context manager to observe latency for the wrapped block."""

    start = time()
    try:
        yield
    finally:
        duration = time() - start
        INFERENCE_LATENCY.observe(duration)
        logger.debug("Inference completed", duration_seconds=duration)


def metrics_response() -> Response:
    """Return a Prometheus metrics response."""

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def configure_tracing(settings: Settings) -> None:
    """Configure OpenTelemetry tracing if an OTLP endpoint is provided."""

    if not settings.otlp_endpoint:
        logger.info("OTLP endpoint not provided; tracing disabled.")
        return
    if not OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry is not installed. Install the 'observability' dependency group to enable tracing."
        )
        return

    resource = Resource.create({"service.name": "polyp-triton-gateway"})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    logger.info("Tracing configured", endpoint=settings.otlp_endpoint)


@contextmanager
def start_span(name: str):
    if OTEL_AVAILABLE and trace is not None:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(name) as span:
            yield span
    else:
        yield None


__all__ = [
    "REQUEST_COUNTER",
    "INFERENCE_LATENCY",
    "HEALTH_GAUGE",
    "track_latency",
    "metrics_response",
    "configure_tracing",
    "start_span",
]
