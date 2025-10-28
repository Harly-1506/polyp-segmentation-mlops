from typing import Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Original filename supplied by the client")
    mask: str = Field(..., description="Binary mask encoded as base64 PNG")
    overlay: str = Field(..., description="Overlay image encoded as base64 PNG")
    coverage: float = Field(..., description="Percentage of pixels marked positive")
    latency_seconds: float = Field(..., description="End-to-end latency for the request")


class HealthResponse(BaseModel):
    status: str
    live: bool
    ready: bool
    model_ready: bool
    details: Optional[dict]


__all__ = ["PredictionResponse", "HealthResponse"]
