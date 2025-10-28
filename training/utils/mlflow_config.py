# training/mlflow_tracker.py
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

import mlflow

load_dotenv("./.env")


class MLflowTracker:
    def __init__(
        self,
        experiment_name: str = "development",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        enable_logging: bool = True,
    ):
        self.enabled = enable_logging
        if not self.enabled:
            return

        self.logger = logging.getLogger(__name__)
        self.MLFLOW_TRACKING_URI = os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5001"
        )

        try:
            mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
            self._verify_mlflow_connection()

            if not experiment_name:
                experiment_name = "Default_" + datetime.now().strftime("%Y%m%d")
            self.experiment_name = experiment_name
            self._setup_experiment()

            run_name = run_name or f"run_{datetime.now().strftime('%H%M%S')}"
            self.run = mlflow.start_run(run_name=run_name)

            default_tags = {
                "project": "End to end segmentation project",
                "environment": (
                    "development"
                    if "localhost" in self.MLFLOW_TRACKING_URI
                    else "production"
                ),
                "author": "Harly",
            }
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)

            self.logger.info(
                f"MLflow tracking initialized at {self.MLFLOW_TRACKING_URI}"
            )

        except Exception as e:
            self.enabled = False
            self.logger.error(f"Failed to initialize MLflow: {str(e)}")
            self.logger.warning("Continuing without MLflow tracking")

    def _verify_mlflow_connection(self):
        try:
            if not mlflow.tracking.get_tracking_uri():
                raise ValueError("MLflow tracking URI not set")
            experiments = mlflow.search_experiments()
            self.logger.debug(
                f"Connected to MLflow. Found {len(experiments)} experiments"
            )
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to MLflow at {self.MLFLOW_TRACKING_URI}: {str(e)}"
            )

    def _setup_experiment(self):
        try:
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            raise RuntimeError(f"Failed to setup experiment: {str(e)}")

    def log_params(self, params: Dict[str, Any]):
        if not self.enabled:
            return
        try:
            mlflow.log_params(params)
        except Exception as e:
            self.logger.error(f"Failed to log params: {str(e)}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if not self.enabled:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {str(e)}")

    def log_artifact(self, path: str):
        if not self.enabled:
            return
        try:
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="checkpoints")
            else:
                self.logger.warning(f"Artifact path not found: {path}")
        except Exception as e:
            self.logger.error(f"Failed to log artifact: {str(e)}")

    def log_model(self, model, artifact_path: str):
        if not self.enabled:
            return
        try:
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            self.logger.error(f"Failed to log model: {str(e)}")

    def end_run(self):
        if not self.enabled:
            return
        try:
            mlflow.end_run()
            self.logger.info("MLflow run completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to end MLflow run: {str(e)}")

    def torch_log_model(self, model, artifact_path: str):
        if not self.enabled:
            return
        try:
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            self.logger.error(f"Failed to log PyTorch model: {str(e)}")
