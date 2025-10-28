import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from utils.logger_config import measure_execution_time, setup_logger

from training.models.unet import UNet


class SegmentationPipeline:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model()
        self.transform = self.get_transforms()

    def load_model(self):
        model = UNet(1)
        checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        return image

    def preprocess(self, image: Image.Image):
        tensor_img = self.transform(image).unsqueeze(0)
        return tensor_img

    @measure_execution_time
    def predict(self, image_tensor: torch.Tensor):
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits)
            mask = probs.squeeze().cpu().numpy()
            if mask.ndim == 3:
                mask = np.argmax(mask, axis=0)
            else:
                mask = (mask > 0.5).astype(np.uint8)
        return mask

    def overlap_mask_on_image(
        self, image: Image.Image, mask: np.ndarray
    ) -> Image.Image:
        image_resized = image.resize(mask.shape[::-1])
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.convert("L")
        image_array = np.array(image_resized)
        mask_array = np.array(mask_img)
        red_mask = np.zeros_like(image_array)
        red_mask[..., 0] = mask_array
        overlapped = np.clip(image_array * 0.7 + red_mask * 0.3, 0, 255).astype(
            np.uint8
        )
        # rezize the overlapped image to match the original image size
        overlapped = Image.fromarray(overlapped)
        overlapped = overlapped.resize(image.size)
        return overlapped

    def show_overlap_only(self, image: Image.Image, prediction: np.ndarray):
        overlapped = self.overlap_mask_on_image(image, prediction)
        plt.figure(figsize=(6, 6))
        plt.imshow(overlapped)
        plt.axis("off")
        plt.title("Overlap Mask")
        plt.tight_layout()
        plt.show()

    def show_image_and_mask(self, image: Image.Image, prediction: np.ndarray):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(prediction)
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    def visualize(self, image: Image.Image, prediction: np.ndarray, model_name: str):
        overlapped = self.overlap_mask_on_image(image, prediction)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(prediction)
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        axes[2].imshow(overlapped)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        out_dir = Path("visualization")
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"{Path(model_name).stem}_overlay.png")
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run segmentation prediction using a trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="training/checkpoints/UNet/Unet81PolypPVT-best.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization of results"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = SegmentationPipeline(args.model)
    image = pipeline.load_image(args.image_path)
    tensor_img = pipeline.preprocess(image)
    prediction = pipeline.predict(tensor_img)
    if args.visualize:
        pipeline.visualize(image, prediction, model_name=args.model)


if __name__ == "__main__":
    main()
