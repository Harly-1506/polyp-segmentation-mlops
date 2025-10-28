import argparse
import json
import logging
import os
import random
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from models.unet import UNet
from torch.autograd import Variable
from trainer.trainer import SegmentationTrainer
from utils.dataset import TestDataset, get_loader
from utils.utils import AvgMeter, clip_gradient

load_dotenv("./.env")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.simplefilter(action="ignore", category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=3, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument(
    "--optimizer",
    type=str,
    default="AdamW",
    choices=["AdamW", "SGD"],
    help="Optimizer to use",
)
parser.add_argument(
    "--augmentation", action="store_true", help="Enable data augmentation"
)
parser.add_argument("--batchsize", type=int, default=1, help="Training batch size")
parser.add_argument("--trainsize", type=int, default=256, help="Training image size")
parser.add_argument("--clip", type=float, default=0.5, help="Gradient clipping value")
parser.add_argument(
    "--decay_rate", type=float, default=0.1, help="Learning rate decay factor"
)
parser.add_argument(
    "--decay_epoch", type=int, default=50, help="Decay learning rate every N epochs"
)
parser.add_argument(
    "--train_path",
    type=str,
    default="./dataset1/TrainDataset/",
    help="Path to training dataset",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="./dataset1/TestDataset/",
    help="Path to testing dataset",
)
parser.add_argument(
    "--train_save",
    type=str,
    default="./training/checkpoints",
    help="Directory to save model checkpoints",
)
parser.add_argument(
    "--model",
    type=str,
    default="UNet",
    choices=["PolypPVT", "UNet"],
    help="Model architecture to use",
)
opt = parser.parse_args()


def create_model(model_name: str):
    if model_name == "UNet":
        return UNet(1)
    else:
        raise ValueError(f"Model '{model_name}' not implemented.")


def main():
    print("\n========== Configuration ==========")
    print(json.dumps(vars(opt), indent=4))
    print("===================================\n")

    model = create_model(opt.model)
    print(
        f"Initialized {opt.model} with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Prepare data loader
    image_root = os.path.join(opt.train_path, "images")
    gt_root = os.path.join(opt.train_path, "masks")
    train_loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        augmentation=opt.augmentation,
    )

    configs = {
        "batchsize": opt.batchsize,
        "lr": opt.lr,
        "num_workers": 0,
        "trainsize": opt.trainsize,
        "clip": opt.clip,
        "epoch": opt.epoch,
        "train_save": os.path.join(opt.train_save, opt.model),
        "optimizer": opt.optimizer,
        "decay_rate": opt.decay_rate,
        "decay_epoch": opt.decay_epoch,
        "model": opt.model,
        "mlflow": os.getenv("MLFLOW_TRACKING_URI") is not None,
        "mlflow_experiment_name": "PolypSegmentation",
        "mlflow_run_name": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
    }

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        test_path=opt.test_path,
        configs=configs,
    )

    print("#" * 20, "Start Training", "#" * 20)
    trainer.train()
    print("#" * 20, "Training Complete", "#" * 20)
    print(f"Best Dice Score: {trainer.best_dice_score:.4f}")


if __name__ == "__main__":
    main()
