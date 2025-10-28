import base64
import io
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from loguru import logger
from PIL import Image

try:
    import tritonclient.http as httpclient
    from tritonclient.utils import InferenceServerException, np_to_triton_dtype
except ImportError:  # checking for tritonclient
    httpclient = None  
    InferenceServerException = Exception 
    np_to_triton_dtype = lambda dtype: str(dtype) 

from .config import Settings


@dataclass
class HealthReport:
    live: bool
    ready: bool
    model_ready: bool
    details: Optional[Dict[str, str]] = None


class BaseSegmentationClient:
    """Abstract base class for inference clients."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def health(self) -> HealthReport:  
        raise NotImplementedError

    def infer(self, image: Image.Image) -> np.ndarray:  
        raise NotImplementedError


class TritonSegmentationClient(BaseSegmentationClient):
    """HTTP client that communicates with NVIDIA Triton."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        if httpclient is None:
            raise RuntimeError(
                "tritonclient is not installed. Install the inference dependency group via `uv pip install -r pyproject.toml -G inference`."
            )
        self._client = httpclient.InferenceServerClient(
            url=settings.resolved_triton_endpoint, concurrency=4, verbose=False
        )
        logger.info(
            "Initialised Triton HTTP client",
            url=settings.resolved_triton_endpoint,
            model=settings.triton_model_name,
            version=settings.resolved_model_version or "latest",
        )

    def health(self) -> HealthReport:
        try:
            version = self.settings.resolved_model_version
            live = self._client.is_server_live()
            ready = self._client.is_server_ready()
            kwargs = {}
            if version is not None:
                kwargs["model_version"] = version
            model_ready = self._client.is_model_ready(
                model_name=self.settings.triton_model_name,
                **kwargs,
            )
            report = HealthReport(live=live, ready=ready, model_ready=model_ready)
        except InferenceServerException as exc:  
            logger.error("Unable to query Triton health", error=str(exc))
            report = HealthReport(live=False, ready=False, model_ready=False, details={"error": str(exc)})
        HEALTH_GAUGE.labels("live").set(float(report.live))
        HEALTH_GAUGE.labels("ready").set(float(report.ready))
        HEALTH_GAUGE.labels("model_ready").set(float(report.model_ready))
        return report

    def infer(self, image: Image.Image) -> np.ndarray:
        array = preprocess(image, self.settings.image_size)
        logger.debug(
            "Prepared tensor",
            shape=array.shape,
            dtype=str(array.dtype),
        )

        inputs = [
            httpclient.InferInput(
                name=self.settings.triton_input_name,
                shape=array.shape,
                datatype=np_to_triton_dtype(array.dtype),
            )
        ]
        inputs[0].set_data_from_numpy(array)

        outputs = [httpclient.InferRequestedOutput(self.settings.triton_output_name)]

        with track_latency():
            version = self.settings.resolved_model_version
            kwargs = {}
            if version is not None:
                kwargs["model_version"] = version
            response = self._client.infer(
                model_name=self.settings.triton_model_name,
                inputs=inputs,
                outputs=outputs,
                timeout=self.settings.request_timeout_seconds * 1000,
                **kwargs,
            )
        mask = response.as_numpy(self.settings.triton_output_name)
        if mask is None:
            raise RuntimeError("Inference response missing output tensor")
        logger.debug(
            "Received mask",
            shape=mask.shape,
            dtype=str(mask.dtype),
        )
        return postprocess(mask, self.settings.confidence_threshold)


class MockSegmentationClient(BaseSegmentationClient):
    """Fallback client for local development when Triton is unavailable."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        logger.warning("Using MockSegmentationClient; no real inference will occur.")

    def health(self) -> HealthReport:
        report = HealthReport(live=True, ready=True, model_ready=True, details={"mock": "true"})
        HEALTH_GAUGE.labels("live").set(1)
        HEALTH_GAUGE.labels("ready").set(1)
        HEALTH_GAUGE.labels("model_ready").set(1)
        return report

    def infer(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(image.convert("L"))
        mask = (array > array.mean()).astype(np.uint8)
        logger.debug("Mock inference completed", mask_shape=str(mask.shape))
        return mask


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((size, size))
    array = np.asarray(resized).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def postprocess(mask: np.ndarray, threshold: float) -> np.ndarray:
    squeezed = np.squeeze(mask)
    if squeezed.ndim == 3:
        squeezed = squeezed[0]
    squeezed = squeezed.astype(np.float32)
    binary = (squeezed >= threshold).astype(np.uint8)
    return binary


def decode_base64_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_mask(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


from .instrumentation import HEALTH_GAUGE, track_latency 

__all__ = [
    "BaseSegmentationClient",
    "TritonSegmentationClient",
    "MockSegmentationClient",
    "HealthReport",
    "preprocess",
    "postprocess",
    "decode_base64_image",
    "encode_mask",
]
