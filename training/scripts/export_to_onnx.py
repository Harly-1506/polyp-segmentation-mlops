"""Export PyTorch checkpoints to ONNX for Triton deployments."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

try:
    from training.models.unet import UNet
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("UNet model definition not found. Ensure training.models is on PYTHONPATH.") from exc


def load_model(checkpoint: Path) -> torch.nn.Module:
    model = UNet(1)
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    model.load_state_dict(payload)
    model.eval()
    return model


def export_to_onnx(checkpoint: Path, output: Path, image_size: int, opset: int, dynamic: bool) -> None:
    model = load_model(checkpoint)
    dummy = torch.randn(1, 3, image_size, image_size)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 1: "classes", 2: "height", 3: "width"},
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"Exported {checkpoint} -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to the PyTorch checkpoint (.pth)")
    parser.add_argument("output", type=Path, help="Target ONNX file path")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size (square).")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for batch/height/width")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_to_onnx(args.checkpoint, args.output, args.image_size, args.opset, args.dynamic)


if __name__ == "__main__":
    main()
