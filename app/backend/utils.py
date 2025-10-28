import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Overlay a binary mask on top of an image."""

    if mask.ndim != 2:
        raise ValueError("Mask must be 2D after post-processing.")

    image_rgb = image.convert("RGB")
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(image_rgb.size)
    mask_array = np.asarray(mask_resized)
    image_array = np.asarray(image_rgb)
    overlay = image_array.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], mask_array)
    blended = (image_array * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def mask_to_base64(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def compute_mask_stats(mask: np.ndarray) -> Tuple[int, float]:
    pixels = int(mask.size)
    coverage = float(mask.sum() / mask.size)
    return pixels, coverage


__all__ = ["overlay_mask", "image_to_base64", "mask_to_base64", "compute_mask_stats"]
