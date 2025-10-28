import io
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import httpx
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BACKEND_URL = "local"
BACKEND_URL = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)


@contextmanager
def _get_client() -> Iterator[httpx.Client]:
    """Return an HTTP client, falling back to an in-process TestClient."""

    if BACKEND_URL == DEFAULT_BACKEND_URL:
        # Ensure the FastAPI app initialises with the mock client so we do not
        # depend on a real Triton server during tests.
        os.environ.setdefault("APP_ENABLE_MOCK", "true")
        from fastapi.testclient import TestClient

        from app.backend.main import app

        with TestClient(app) as client:  # type: ignore[assignment]
            yield client  # type: ignore[misc]
    else:
        with httpx.Client(base_url=BACKEND_URL) as client:
            yield client


def run() -> None:
    image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    with _get_client() as client:
        health = client.get("/healthz", timeout=5)
        print("Health:", health.json())

        response = client.post(
            "/predict",
            files={"file": ("synthetic.png", buffer.getvalue(), "image/png")},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        print("Prediction keys:", payload.keys())


if __name__ == "__main__":
    run()
