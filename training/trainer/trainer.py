import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.dataset import TestDataset, get_loader
from utils.mlflow_config import MLflowTracker
from utils.utils import AvgMeter, clip_gradient


class SegmentationTrainer:
    def __init__(self, model, train_loader, test_path, configs):
        self.train_loader = train_loader
        self.test_path = test_path
        self.configs = configs

        self.batch_size = configs["batchsize"]
        self.learning_rate = configs["lr"]
        self.num_workers = configs["num_workers"]
        self.trainsize = configs["trainsize"]
        self.clip = configs["clip"]
        self.max_epoch_num = configs["epoch"]
        self.model_name = configs["model"]
        self.use_mlflow = configs.get("mlflow", False)
        self.MLFLOW_TRACKING_URI = configs.get("MLFLOW_TRACKING_URI", None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",  # Monitor dice score (higher is better)
            factor=configs.get("decay_rate", 0.1),
            patience=configs.get("scheduler_patience", 5),
            # verbose=True
        )
        # Metrics tracking
        self.train_loss_list = []
        self.dice_scores = {
            "CVC-300": [],
            "CVC-ClinicDB": [],
            "Kvasir": [],
            "CVC-ColonDB": [],
            "ETIS-LaribPolypDB": [],
            "test": [],
        }
        self.best_dice_score = 0.0

        # Create save directory
        self.save_path = configs["train_save"]
        os.makedirs(self.save_path, exist_ok=True)

        # Initialize logging
        # logging.basicConfig(filename=os.path.join(self.save_path, 'training.log'), level=logging.INFO)

        self.use_mlflow = configs.get("mlflow", False) and (self.world_rank == 0)

        if self.use_mlflow:
            self.mlflow = MLflowTracker(
                experiment_name=configs.get(
                    "mlflow_experiment_name", "PolypSegmentation"
                ),
                run_name=configs.get("mlflow_run_name", "TrainingRun"),
                tags=configs.get(
                    "mlflow_tags",
                    {
                        "project": "PolypSegmentation",
                        "environment": "development",
                        "author": "Harly",
                    },
                ),
            )
            print(f"MLflow tracking enabled at {self.tracking_uri}")
            self.mlflow.log_params(self.configs)
        else:
            print("MLflow tracking disabled")

    def structure_loss(self, pred, mask):
        """Compute the structure loss combining weighted binary cross-entropy and weighted IoU."""
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = (pred * mask * weit).sum(dim=(2, 3))
        union = (pred + mask * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

    def evaluate(self, dataset_name):
        """Evaluates model on a specific dataset and returns average Dice score."""
        self.model.eval()
        image_root = os.path.join(
            self.test_path if dataset_name != "test" else self.test_path,
            dataset_name,
            "images",
        )
        mask_root = os.path.join(
            self.test_path if dataset_name != "test" else self.test_path,
            dataset_name,
            "masks",
        )

        test_loader = TestDataset(image_root, mask_root, 352)
        dice_total = 0.0
        num_samples = len(os.listdir(mask_root))

        for _ in range(num_samples):
            image, mask, _ = test_loader.load_data()
            mask = np.asarray(mask, np.float32)
            mask /= mask.max() + 1e-8
            image = image.to(self.device)

            with torch.no_grad():
                res = self.model(image)
                res = F.interpolate(
                    res, size=mask.shape, mode="bilinear", align_corners=False
                )
                res = torch.sigmoid(res).cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            input_flat = res.flatten()
            target_flat = mask.flatten()
            intersection = (input_flat * target_flat).sum()
            dice = (2 * intersection + 1) / (input_flat.sum() + target_flat.sum() + 1)
            dice_total += dice

        return dice_total / num_samples

    def train_epoch(self, epoch):
        """Train for one epoch with multi-scale inputs"""
        self.model.train()
        size_rates = [0.75, 1, 1.25]
        loss_record = AvgMeter()

        for i, (images, masks) in tqdm.tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            leave=True,
            colour="blue",
            desc=f"Epoch {epoch}",
            bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            for rate in size_rates:
                self.optimizer.zero_grad()
                images_var = Variable(images).to(self.device)
                masks_var = Variable(masks).to(self.device)

                # Rescale images and masks
                trainsize = int(round(self.trainsize * rate / 32) * 32)
                if rate != 1:
                    images_var = F.interpolate(
                        images_var,
                        size=(trainsize, trainsize),
                        mode="bilinear",
                        align_corners=True,
                    )
                    masks_var = F.interpolate(
                        masks_var,
                        size=(trainsize, trainsize),
                        mode="bilinear",
                        align_corners=True,
                    )

                preds = self.model(images_var)
                loss = self.structure_loss(preds, masks_var)
                loss.backward()
                clip_gradient(self.optimizer, self.clip)
                self.optimizer.step()

                if rate == 1:
                    loss_record.update(loss.item(), self.batch_size)

            if i % 20 == 0 or i == len(self.train_loader):
                print(
                    "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}".format(
                        datetime.now(),
                        epoch,
                        self.max_epoch_num,
                        i,
                        len(self.train_loader),
                        loss_record.show(),
                    )
                )

        self.train_loss_list.append(loss_record.avg)

        # Save checkpoint
        # torch.save(self.model.state_dict(), os.path.join(self.save_path, f'{epoch}PolypPVT.pth'))

    def validate(self, epoch):
        """Evaluate model on all test datasets"""
        for dataset in self.dice_scores.keys():
            dice = self.evaluate(dataset)
            if self.use_mlflow:
                self.mlflow.log_metric(f"{dataset}_dice_score", dice, step=epoch)
            # logging.info(f'epoch: {epoch}, dataset: {dataset}, dice: {dice:.4f}')
            print(f"{dataset}: {dice:.4f}")
            self.dice_scores[dataset].append(dice)

        current_dice = self.dice_scores["test"][-1]
        if current_dice > self.best_dice_score:
            self.best_dice_score = current_dice
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_path, f"{self.model_name}.pth"),
            )
            # torch.save(self.model.state_dict(), os.path.join(self.save_path, f'{epoch}PolypPVT-best_dice_score.pth'))
            print("### Best model updated:", self.best_dice_score)
            # logging.info(f'### Best model updated: {self.best_dice_score}')

    def train(self):
        """Main training loop with optional MLflow tracking"""
        # MLflow context manager if enabled
        if self.use_mlflow:
            run = self.mlflow.start_run(run_name=self.configs["mlflow_run_name"])
            print(f"MLflow Run ID: {run.info.run_id}")
            self.mlflow.log_params(self.configs)
            self.mlflow.pytorch.log_model(self.model, "model")
        try:
            for epoch in range(1, self.max_epoch_num + 1):
                self.current_epoch_num = epoch

                # Train for one epoch
                self.train_epoch(epoch)
                self.validate(epoch)
                current_dice = self.dice_scores["test"][-1]
                self.scheduler.step(current_dice)

                # Log learning rate if MLflow is enabled
                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.use_mlflow:
                    self.mlflow.log_metric("learning_rate", current_lr, step=epoch)
                    self.mlflow.log_metric(
                        "train_loss", self.train_loss_list[-1], step=epoch
                    )
                    self.mlflow.log_metric("dice_score", current_dice, step=epoch)
                print(f"Epoch {epoch} - Current LR: {current_lr:.2e}")
                # logging.info(f"Epoch {epoch} - Learning Rate: {current_lr:.2e}")

        except KeyboardInterrupt:
            print("Training interrupted by user")
            if self.use_mlflow:
                self.mlflow.log_param("status", "interrupted")

            if self.use_mlflow:
                self.mlflow.log_param("status", "completed")
                self.mlflow.log_metric("final_best_dice_score", self.best_dice_score)

            print("\n---------------------- TRAINING SUMMARY -----------------------")
            print(f" Best Dice Score: {self.best_dice_score:.4f}")
            print("--------------------------------------------------------------")
