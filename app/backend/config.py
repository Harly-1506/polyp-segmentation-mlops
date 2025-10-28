from functools import lru_cache
from typing import Optional
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables.

    Environment variables are prefixed with ``APP_`` to avoid collisions.
    ``APP_TRITON_URL=localhost:8000`` becomes ``triton_url`` for example, while
    `PREDICTOR_HOST`/`PREDICTOR_PORT` can be supplied without the prefix when
    running in Kubernetes to build the Triton URL dynamically.
    """

    triton_url: str = "localhost:8000" # The URL of the Triton Inference Server.
    triton_grpc_url: Optional[str] = None
    triton_model_name: str = "polyp-segmentation"
    triton_model_version: Optional[str] = None
    triton_input_name: str = "input"
    triton_output_name: str = "output"
    predictor_host: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("PREDICTOR_HOST", "APP_PREDICTOR_HOST"),
    )
    predictor_port: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("PREDICTOR_PORT", "APP_PREDICTOR_PORT"),
    )
    image_size: int = 256
    confidence_threshold: float = 0.5

    request_timeout_seconds: int = 60
    enable_mock: bool = False

    log_level: str = "INFO"

    prometheus_namespace: str = "polyp_service"
    otlp_endpoint: Optional[str] = None

    class Config:
        env_prefix = "APP_"

    @model_validator(mode="after")
    def _apply_predictor_host(self) -> "Settings":
        """If PREDICTOR_HOST/PORT are provided, compose the Triton URL from them."""

        if self.predictor_host and self.predictor_port:
            host = self.predictor_host.strip()
            if "://" in host:
                host = host.split("://", 1)[1]
            host = host.rstrip("/")
            self.triton_url = f"http://{host}:{self.predictor_port}"
        elif self.predictor_host or self.predictor_port:
            raise ValueError("Both PREDICTOR_HOST and PREDICTOR_PORT must be provided together.")
        return self

    @property
    def resolved_triton_url(self) -> str:
        """Return a Triton URL that always includes the HTTP scheme."""
        url = self.triton_url.strip()
        if not url:
            raise ValueError("APP_TRITON_URL must not be empty")
        if "://" not in url:
            url = f"http://{url}"
        parsed = urlparse(url)
        netloc = parsed.netloc or parsed.path
        if not netloc:
            raise ValueError("APP_TRITON_URL is invalid; expected host:port")
        scheme = parsed.scheme or "http"
        return f"{scheme}://{netloc}".rstrip("/")

    @property
    def resolved_triton_endpoint(self) -> str:
        """Return a Triton endpoint in ``host:port`` format for client libraries."""

        parsed = urlparse(self.resolved_triton_url)
        endpoint = parsed.netloc or parsed.path
        if not endpoint:
            raise ValueError("APP_TRITON_URL is invalid; expected host:port")
        return endpoint
    
    
    @property
    def resolved_model_version(self) -> Optional[str]:
        """Return the Triton model version as a string when provided."""

        if self.triton_model_version in (None, ""):
            return None
        return str(self.triton_model_version)



@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of :class:`Settings`."""

    return Settings()


__all__ = ["Settings", "get_settings"]
