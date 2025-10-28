import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable

import ray
from training.utils.dataset import TestDataset
from training.utils.mlflow_config import MLflowTracker
from training.utils.utils import AvgMeter, clip_gradient


class SegmentationTrainer:
    def __init__(self, model, train_loader, test_path, configs, world_rank):
        self.logger = logging.getLogger(__name__)
        self.train_loader = train_loader
        self.test_path = test_path
        self.configs = configs

        self.batch_size = configs["batchsize"]
        self.learning_rate = configs["lr"]
        self.trainsize = configs["trainsize"]
        self.clip = configs["clip"]
        self.max_epoch_num = configs["epoch"]
        self.model_name = configs["model"]
        self.mlflow_configs = configs.get("mlflow", {})
        self.use_mlflow = self.mlflow_configs.get("enable", False)
        self.MLFLOW_TRACKING_URI = configs.get(
            "MLFLOW_TRACKING_URI", None
        ) or self.mlflow_configs.get("tracking_uri", None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_rank = world_rank

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=configs.get("decay_rate", 0.1),
            patience=configs.get("scheduler_patience", 5),
        )

        self.train_loss_list = []
        self.dice_scores = {
            ds: []
            for ds in [
                "CVC-300",
                "CVC-ClinicDB",
                "Kvasir",
                "CVC-ColonDB",
                "ETIS-LaribPolypDB",
                "test",
            ]
        }

        self.best_dice_score = 0.0

        self.save_path = configs["train_save"]
        os.makedirs(self.save_path, exist_ok=True)

        run_name = self.mlflow_configs.get("run_name") or "TrainingRun"

        # MLflow tracker
        if self.use_mlflow and self.world_rank == 0 and self.MLFLOW_TRACKING_URI:
            self.mlflow = MLflowTracker(
                experiment_name=self.mlflow_configs.get(
                    "experiment_name", "PolypSegmentation"
                ),
                run_name=run_name,
                tags=self.mlflow_configs.get("tags")
                or self.mlflow_configs.get(
                    "mlflow_tags",
                    {
                        "project": "PolypSegmentation",
                        "environment": "development",
                        "author": "Harly",
                    },
                ),
            )
            self.logger.info("MLflow tracking enabled at %s", self.MLFLOW_TRACKING_URI)
            self.mlflow.log_params(self.configs)
            self._log_system_info()
        else:
            self.mlflow = None
            if self.use_mlflow and self.world_rank == 0:
                self.logger.warning(
                    "MLflow tracking requested but MLFLOW_TRACKING_URI is not set; disabling logging"
                )
            else:
                self.logger.info("MLflow tracking disabled")

    def _log_system_info(self):
        """Log system/GPU info to MLflow."""
        if not self.mlflow:
            return
        if torch.cuda.is_available():
            self.mlflow.log_params(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "cuda_version": torch.version.cuda,
                }
            )

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pred = torch.sigmoid(pred)
        inter = (pred * mask * weit).sum(dim=(2, 3))
        union = (pred + mask * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def _compute_dice(self, pred, mask):
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        input_flat = pred.flatten()
        target_flat = mask.flatten()
        intersection = (input_flat * target_flat).sum()
        return (2 * intersection + 1) / (input_flat.sum() + target_flat.sum() + 1)

    def evaluate(self, dataset_name, epoch):
        self.model.eval()
        image_root = os.path.join(self.test_path, dataset_name, "images")
        mask_root = os.path.join(self.test_path, dataset_name, "masks")
        test_loader = TestDataset(image_root, mask_root, 352)

        dice_total = 0.0
        num_samples = len(os.listdir(mask_root))

        for _ in tqdm.tqdm(
            range(num_samples),
            desc=f"Evaluating {dataset_name} epoch {epoch}",
            colour="green",
        ):
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

            dice = self._compute_dice(res, mask)
            dice_total += dice

        return dice_total / num_samples

    def train_epoch(self, epoch):
        self.model.train()
        size_rates = [0.75, 1, 1.25]
        loss_record = AvgMeter()

        for i, (images, masks) in tqdm.tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            colour="blue",
            desc=f"Epoch {epoch}",
        ):
            for rate in size_rates:
                self.optimizer.zero_grad()
                images_var = Variable(images).to(self.device)
                masks_var = Variable(masks).to(self.device)

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

            if i % 20 == 0 and self.use_mlflow and self.world_rank == 0 and self.mlflow:
                self.mlflow.log_metrics(
                    {"batch_loss": loss_record.avg},
                    step=(epoch * len(self.train_loader) + i),
                )

        self.train_loss_list.append(loss_record.avg)

    def validate(self, epoch):
        for dataset in self.dice_scores.keys():
            dice = self.evaluate(dataset, epoch)
            self.dice_scores[dataset].append(dice)
            if self.use_mlflow and self.world_rank == 0 and self.mlflow:
                self.mlflow.log_metrics({f"{dataset}_dice_score": dice}, step=epoch)

        current_dice = self.dice_scores["test"][-1]
        if current_dice > self.best_dice_score:
            self.best_dice_score = current_dice
            save_path = os.path.join(
                self.save_path,
                f'{self.model_name}_{self.configs.get("mlflow_run_name", "TrainingRun")}_best.pth',
            )
            torch.save(self.model.state_dict(), save_path)
            if self.use_mlflow and self.mlflow:
                self.mlflow.log_artifact(save_path)
            self.logger.info("### Best model updated: %.4f", self.best_dice_score)

    def train(self):
        start_time = time.time()
        try:
            for epoch in range(1, self.max_epoch_num + 1):
                self.train_epoch(epoch)
                current_dice = 0.0
                if self.world_rank == 0:
                    self.validate(epoch)
                    current_dice = self.dice_scores["test"][-1]
                    self.scheduler.step(current_dice)
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    if self.use_mlflow and self.mlflow:
                        self.mlflow.log_metrics(
                            {"learning_rate": current_lr}, step=epoch
                        )
                        self.mlflow.log_metrics(
                            {"train_loss": self.train_loss_list[-1]}, step=epoch
                        )
                        self.mlflow.log_metrics(
                            {"dice_score": current_dice}, step=epoch
                        )

                    ray.train.report(
                        {
                            "epoch": epoch,
                            "dice": current_dice,
                            "best_dice_score": self.best_dice_score,
                            "train_loss": self.train_loss_list[-1],
                        }
                    )
                else:
                    ray.train.report(
                        {
                            "epoch": epoch,
                            "final_best_dice_score": self.best_dice_score,
                            "train_loss": self.train_loss_list[-1],
                        }
                    )
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted")
        finally:
            total_time = time.time() - start_time
            if self.use_mlflow and self.world_rank == 0 and self.mlflow:
                self.mlflow.log_params({"total_training_time_sec": total_time})
                self.mlflow.end_run()
