import io
import sys
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from PIL import Image

from .clients import BaseSegmentationClient, MockSegmentationClient, TritonSegmentationClient
from .config import Settings, get_settings
from .instrumentation import REQUEST_COUNTER, configure_tracing, metrics_response, start_span
from .schemas import HealthResponse, PredictionResponse
from .utils import compute_mask_stats, image_to_base64, mask_to_base64, overlay_mask

app = FastAPI(title="Polyp Segmentation Gateway", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)

settings: Settings = get_settings()
client: Optional[BaseSegmentationClient] = None


@app.on_event("startup")
async def startup_event() -> None:
    global client  
    logger.remove()
    logger.add(sys.stdout, level=settings.log_level, colorize=True, enqueue=True)
    configure_tracing(settings)
    client = _build_client(settings)
    report = client.health()
    logger.info(
        "Connecting Triton",
        url=settings.resolved_triton_url,
        model=settings.triton_model_name,
        version=settings.resolved_model_version or "latest",
    )
    logger.info(
        "Service initialised",
        live=report.live,
        ready=report.ready,
        model_ready=report.model_ready,
        details=report.details,
    )


def _build_client(settings: Settings) -> BaseSegmentationClient:
    if settings.enable_mock:
        return MockSegmentationClient(settings)
    return TritonSegmentationClient(settings)


@app.middleware("http")
async def add_request_logging(request: Request, call_next): 
    start_time = time.time()
    with start_span(request.url.path):
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            logger.debug(
                "Request handled",
                path=request.url.path,
                status=response.status_code,
                duration=duration,
            )
            return response
        except Exception as exc: 
            duration = time.time() - start_time
            logger.exception("Unhandled exception", duration=duration)
            raise exc


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    if client is None:
        raise HTTPException(status_code=503, detail="Client not initialised")
    report = client.health()
    status = "ok" if report.ready and report.model_ready else "degraded"
    return HealthResponse(
        status=status,
        live=report.live,
        ready=report.ready,
        model_ready=report.model_ready,
        details=report.details,
    )


@app.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    return await healthz()


@app.get("/metrics")
async def metrics():
    return metrics_response()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if client is None:
        raise HTTPException(status_code=503, detail="Client not initialised")

    REQUEST_COUNTER.labels(endpoint="predict", status="received").inc()
    start = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        mask = client.infer(image)
        overlay = overlay_mask(image, mask)
        latency = time.time() - start
        coverage = compute_mask_stats(mask)[1]
        logger.debug(
            "Inference succeeded",
            filename=file.filename,
            latency_seconds=latency,
            coverage=coverage,
        )
        REQUEST_COUNTER.labels(endpoint="predict", status="success").inc()
        return PredictionResponse(
            filename=file.filename or "uploaded-image.png",
            mask=mask_to_base64(mask),
            overlay=image_to_base64(overlay),
            coverage=coverage,
            latency_seconds=latency,
        )
    except Exception as exc:  
        REQUEST_COUNTER.labels(endpoint="predict", status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
async def index() -> dict:
    return {"status": "ok", "message": "Polyp segmentation API"}


__all__ = ["app"]
