# import argparse
# import json
# import os
# from pathlib import Path
# from typing import Any, Dict

# import yaml
# from kfp import compiler, dsl
# from kfp.dsl import Metrics, Model

# BASE_IMAGE = "harly1506/polyp-mlops:kfpv2"
# @dsl.component(base_image=BASE_IMAGE)
# def download_dataset(
#     dataset_artifact: dsl.Output[dsl.Dataset],
#     dataset_info_artifact: dsl.Output[dsl.Artifact],
#     kube_config_path: str = "training/configs/kube_configs.yaml",
#     env_path: str = ".env",
#     minio_endpoint: str = "",
#     minio_access_key: str = "",
#     minio_secret_key: str = "",
#     minio_secure: bool = False,
# ) -> None:
#     """Download the training and testing dataset from MinIO."""
#     import logging
#     import platform
#     import sys
#     from pathlib import Path as _Path

#     import boto3
#     from botocore.client import Config as BotoConfig
#     from dotenv import dotenv_values

#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("download_dataset")

#     logger.info(
#         "Environment: python=%s platform=%s",
#         sys.version.replace("\n", " "),
#         platform.platform(),
#     )
#     #check configs file exists
#     def _load_yaml_local(path: str) -> Dict[str, Any]:
#         from pathlib import Path as __Path

#         with __Path(path).open("r", encoding="utf-8") as fp:
#             return yaml.safe_load(fp) or {}

#     kube_cfg = _load_yaml_local(kube_config_path)
#     bucket = kube_cfg.get("bucket")
#     prefix = kube_cfg.get("dataset_prefix", "").rstrip("/")
#     mlflow_cfg = kube_cfg.get("mlflow", {})
#     if not bucket or not prefix:
#         raise ValueError("bucket and dataset_prefix must be provided in kube_configs.yaml")

#     dataset_root = _Path(dataset_artifact.path)
#     dataset_root.mkdir(parents=True, exist_ok=True)

#     env_values = {}
#     env_file = _Path(env_path)
#     if env_file.exists():
#         env_values = {k: str(v) for k, v in dotenv_values(env_file).items() if v is not None}
#         for key, value in env_values.items():
#             os.environ.setdefault(key, value)
#         logger.info("Loaded %d entries from %s", len(env_values), env_file)
#     else:
#         logger.warning("Env file %s not found; relying on provided parameters", env_file)

#     access_key = (
#         minio_access_key
#         or env_values.get("AWS_ACCESS_KEY_ID")
#         or env_values.get("MINIO_ROOT_USER")
#     )
#     secret_key = (
#         minio_secret_key
#         or env_values.get("AWS_SECRET_ACCESS_KEY")
#         or env_values.get("MINIO_ROOT_PASSWORD")
#     )
#     endpoint = (
#         minio_endpoint
#         or env_values.get("MINIO_ENDPOINT")
#         or mlflow_cfg.get("s3_endpoint_url")
#         or mlflow_cfg.get("s3_endpoint")
#     )

#     if endpoint and not endpoint.startswith("http"):
#         scheme = "https" if minio_secure else "http"
#         endpoint = f"{scheme}://{endpoint.lstrip('/') }"

#     if not endpoint:
#         raise ValueError("MinIO endpoint must be supplied either via parameters or .env file")
#     if not access_key or not secret_key:
#         raise ValueError("MinIO credentials are required")

#     session = boto3.session.Session(
#         aws_access_key_id=access_key,
#         aws_secret_access_key=secret_key,
#     )
#     s3 = session.resource(
#         "s3",
#         endpoint_url=endpoint,
#         config=BotoConfig(signature_version="s3v4"),
#         region_name=session.region_name or "us-east-1",
#     )
#     bucket_obj = s3.Bucket(bucket)

#     def _download_prefix(source_prefix: str, target_root: _Path) -> int:
#         files_downloaded = 0
#         logger.info("Downloading s3://%s/%s -> %s", bucket, source_prefix, target_root)
#         for obj in bucket_obj.objects.filter(Prefix=source_prefix):
#             key = obj.key
#             if key.endswith("/"):
#                 continue
#             relative_key = key[len(source_prefix) :].lstrip("/")
#             destination = target_root / relative_key
#             destination.parent.mkdir(parents=True, exist_ok=True)
#             bucket_obj.download_file(key, str(destination))
#             files_downloaded += 1
#         logger.info("Downloaded %d files from prefix %s", files_downloaded, source_prefix)
#         return files_downloaded

#     train_candidates = [
#         f"{prefix}/TrainDataset",
#         f"{prefix}/Traindataset",
#     ]
#     test_prefix = f"{prefix}/TestDataset"

#     downloaded = 0
#     train_target = dataset_root / "TrainDataset"
#     for candidate in train_candidates:
#         downloaded += _download_prefix(candidate, train_target)
#         if downloaded:
#             break
#     if not downloaded:
#         raise FileNotFoundError("No training files found under any TrainDataset prefix")

#     test_target = dataset_root / "TestDataset"
#     test_downloaded = _download_prefix(test_prefix, test_target)
#     if not test_downloaded:
#         logger.warning("No testing files found under prefix %s", test_prefix)

#     expected_train_dirs = ["images", "masks"]
#     missing_train_dirs = [
#         subdir for subdir in expected_train_dirs if not (train_target / subdir).exists()
#     ]
#     if missing_train_dirs:
#         raise FileNotFoundError(
#             f"Missing expected training sub-directories under {train_target}: {missing_train_dirs}"
#         )

#     test_datasets = []
#     if test_target.exists():
#         for dataset_dir in sorted(test_target.iterdir()):
#             if not dataset_dir.is_dir():
#                 continue
#             dataset_name = dataset_dir.name
#             required = [dataset_dir / "images", dataset_dir / "masks"]
#             if not all(path.exists() for path in required):
#                 logger.warning(
#                     "Skipping dataset %s due to missing images/masks directories", dataset_name
#                 )
#                 continue
#             test_datasets.append(dataset_name)
#     if not test_datasets:
#         logger.warning("No test datasets discovered beneath %s", test_target)
#     else:
#         logger.info("Discovered %d test datasets: %s", len(test_datasets), test_datasets)

#     logger.info("Dataset root contents:")
#     for path in sorted(dataset_root.glob("**/*")):
#         logger.info(" - %s", path)

#     info = {
#         "bucket": bucket,
#         "prefix": prefix,
#         "root_dir": str(dataset_root),
#         "train_path": str((dataset_root / "TrainDataset").resolve()),
#         "test_path": str((dataset_root / "TestDataset").resolve()),
#         "test_datasets": test_datasets,
#     }
#     with open(dataset_info_artifact.path, "w", encoding="utf-8") as fp:
#         json.dump(info, fp, indent=2)
#     logger.info("Dataset metadata written to %s", dataset_info_artifact.path)


# @dsl.component(base_image=BASE_IMAGE)
# def ray_tune_component(
#     dataset_artifact: dsl.Input[dsl.Dataset],
#     dataset_info_artifact: dsl.Input[dsl.Artifact],
#     best_config_artifact: dsl.Output[dsl.Artifact],
#     kube_config_path: str = "training/configs/kube_configs.yaml",
#     profile: str = "",
#     env_path: str = ".env",
#     ray_address: str = "",
#     ray_namespace: str = "",
#     ray_storage_path: str = "",
#     mlflow_tracking_uri: str = "",
#     mlflow_s3_endpoint: str = "",
# ) -> None:
#     """Run Ray Tune to obtain the best configuration."""
#     import logging
#     import platform
#     import sys
#     from pathlib import Path as _Path

#     import ray
#     import torch
#     from dotenv import dotenv_values

#     from training import ray_main
#     from training.configs.load_configs import load_config

#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("ray_tune")

#     logger.info(
#         "Environment: python=%s torch=%s ray=%s platform=%s",
#         sys.version.replace("\n", " "),
#         torch.__version__,
#         ray.__version__,
#         platform.platform(),
#     )

#     env_file = _Path(env_path)
#     if env_file.exists():
#         env_values = {k: str(v) for k, v in dotenv_values(env_file).items() if v is not None}
#         for key, value in env_values.items():
#             os.environ.setdefault(key, value)
#         logger.info("Loaded %d env vars from %s", len(env_values), env_file)

#     dataset_info = json.loads(_Path(dataset_info_artifact.path).read_text(encoding="utf-8"))
#     logger.info("Resolved dataset info: %s", json.dumps(dataset_info, indent=2))

#     configs = load_config(kube_config_path)
#     if profile:
#         configs["profile"] = profile

#     configs["train_path"] = dataset_info["train_path"]
#     configs["test_path"] = dataset_info["test_path"]
#     dataset_root = _Path(dataset_artifact.path)
#     tune_save_dir = (dataset_root / "artifacts").resolve()
#     tune_save_dir.mkdir(parents=True, exist_ok=True)
#     configs["train_save"] = str(tune_save_dir)

#     def _merge_mlflow_config_local(
#         configs: Dict[str, Any], tracking_uri: str, s3_endpoint: str
#     ) -> None:
#         mlflow_cfg = configs.setdefault("mlflow", {})
#         if "s3_endpoint_url" in mlflow_cfg and "s3_endpoint" not in mlflow_cfg:
#             mlflow_cfg["s3_endpoint"] = mlflow_cfg.get("s3_endpoint_url")
#         if tracking_uri:
#             mlflow_cfg["tracking_uri"] = tracking_uri
#         if s3_endpoint:
#             mlflow_cfg["s3_endpoint"] = s3_endpoint

#     _merge_mlflow_config_local(configs, mlflow_tracking_uri, mlflow_s3_endpoint)

#     mlflow_cfg = configs.setdefault("mlflow", {})
#     run_name = mlflow_cfg.get("run_name")
#     if not run_name or run_name == "auto":
#         run_name = ray_main.create_run_name(configs)
#         mlflow_cfg["run_name"] = run_name
#     configs["mlflow_run_name"] = run_name

#     ray_configs = dict(configs.get("ray") or {})
#     if ray_address:
#         ray_configs["address"] = ray_address
#     if ray_namespace:
#         ray_configs["namespace"] = ray_namespace
#     if ray_storage_path:
#         ray_configs["storage_path"] = ray_storage_path
#     configs["ray"] = ray_configs

#     ray_main.setup_logging(str(configs.get("log_level", "INFO")))
#     ray_main.resolve_mlflow_env(configs)
#     ray_main.seed_everything(int(configs.get("seed", 1234)))

#     ray_main._override_ray_from_env(ray_configs)
#     storage_path = ray_main.resolve_storage_path(configs)
#     scaling_config = ray_main.build_scaling_config(configs)

#     init_kwargs = {
#         "address": ray_configs.get("address"),
#         "namespace": ray_configs.get("namespace"),
#         "runtime_env": ray_configs.get("runtime_env"),
#         "ignore_reinit_error": True,
#         "local_mode": bool(ray_configs.get("local_mode", False)),
#     }
#     init_kwargs = {k: v for k, v in init_kwargs.items() if v}

#     logger.info("Initialising Ray with: %s", json.dumps(init_kwargs, indent=2))
#     ray.init(**init_kwargs)

#     try:
#         best_config = ray_main.run_tuning(configs, scaling_config, storage_path)
#         logger.info("Best hyperparameters: %s", json.dumps(best_config, indent=2))
#         with open(best_config_artifact.path, "w", encoding="utf-8") as fp:
#             json.dump(best_config, fp, indent=2)
#     finally:
#         ray.shutdown()


# @dsl.component(base_image=BASE_IMAGE)
# def ray_train_component(
#     dataset_artifact: dsl.Input[dsl.Dataset],
#     dataset_info_artifact: dsl.Input[dsl.Artifact],
#     best_config_artifact: dsl.Input[dsl.Artifact],
#     model_artifact: dsl.Output[Model],
#     training_summary_artifact: dsl.Output[dsl.Artifact],
#     kube_config_path: str = "training/configs/kube_configs.yaml",
#     profile: str = "",
#     env_path: str = ".env",
#     ray_address: str = "",
#     ray_namespace: str = "",
#     ray_storage_path: str = "",
#     mlflow_tracking_uri: str = "",
#     mlflow_s3_endpoint: str = "",
# ) -> None:
#     """Run the final Ray training job using the best configuration."""
#     import logging
#     import platform
#     import sys
#     from pathlib import Path as _Path

#     import ray
#     import torch
#     from dotenv import dotenv_values

#     from training import ray_main
#     from training.configs.load_configs import load_config

#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("ray_train")

#     logger.info(
#         "Environment: python=%s torch=%s ray=%s platform=%s",
#         sys.version.replace("\n", " "),
#         torch.__version__,
#         ray.__version__,
#         platform.platform(),
#     )

#     env_file = _Path(env_path)
#     if env_file.exists():
#         env_values = {k: str(v) for k, v in dotenv_values(env_file).items() if v is not None}
#         for key, value in env_values.items():
#             os.environ.setdefault(key, value)
#         logger.info("Loaded %d env vars from %s", len(env_values), env_file)

#     dataset_info = json.loads(_Path(dataset_info_artifact.path).read_text(encoding="utf-8"))
#     logger.info("Resolved dataset info: %s", json.dumps(dataset_info, indent=2))
#     best_config = json.loads(_Path(best_config_artifact.path).read_text(encoding="utf-8"))
#     logger.info("Loaded best config overrides: %s", json.dumps(best_config, indent=2))

#     configs = load_config(kube_config_path)
#     if profile:
#         configs["profile"] = profile

#     configs["train_path"] = dataset_info["train_path"]
#     configs["test_path"] = dataset_info["test_path"]
#     dataset_root = _Path(dataset_artifact.path)
#     model_dir = _Path(model_artifact.path).resolve()
#     model_dir.mkdir(parents=True, exist_ok=True)
#     configs["train_save"] = str(model_dir)

#     def _merge_mlflow_config_local(
#         configs: Dict[str, Any], tracking_uri: str, s3_endpoint: str
#     ) -> None:
#         mlflow_cfg = configs.setdefault("mlflow", {})
#         if "s3_endpoint_url" in mlflow_cfg and "s3_endpoint" not in mlflow_cfg:
#             mlflow_cfg["s3_endpoint"] = mlflow_cfg.get("s3_endpoint_url")
#         if tracking_uri:
#             mlflow_cfg["tracking_uri"] = tracking_uri
#         if s3_endpoint:
#             mlflow_cfg["s3_endpoint"] = s3_endpoint

#     _merge_mlflow_config_local(configs, mlflow_tracking_uri, mlflow_s3_endpoint)

#     mlflow_cfg = configs.setdefault("mlflow", {})
#     run_name = mlflow_cfg.get("run_name")
#     if not run_name or run_name == "auto":
#         run_name = ray_main.create_run_name(configs)
#     final_run_name = f"{run_name}_final"
#     mlflow_cfg["run_name"] = final_run_name
#     configs["mlflow_run_name"] = final_run_name
#     configs["is_final_training"] = True

#     ray_main._deep_update(configs, best_config)

#     ray_configs = dict(configs.get("ray") or {})
#     if ray_address:
#         ray_configs["address"] = ray_address
#     if ray_namespace:
#         ray_configs["namespace"] = ray_namespace
#     if ray_storage_path:
#         ray_configs["storage_path"] = ray_storage_path
#     configs["ray"] = ray_configs

#     ray_main.setup_logging(str(configs.get("log_level", "INFO")))
#     ray_main.resolve_mlflow_env(configs)
#     ray_main.seed_everything(int(configs.get("seed", 1234)))

#     ray_main._override_ray_from_env(ray_configs)
#     storage_path = ray_main.resolve_storage_path(configs)
#     scaling_config = ray_main.build_scaling_config(configs)

#     init_kwargs = {
#         "address": ray_configs.get("address"),
#         "namespace": ray_configs.get("namespace"),
#         "runtime_env": ray_configs.get("runtime_env"),
#         "ignore_reinit_error": True,
#         "local_mode": bool(ray_configs.get("local_mode", False)),
#     }
#     init_kwargs = {k: v for k, v in init_kwargs.items() if v}

#     logger.info("Initialising Ray with: %s", json.dumps(init_kwargs, indent=2))
#     ray.init(**init_kwargs)

#     try:
#         result = ray_main.run_final_training(configs, scaling_config, storage_path)
#         raw_metrics = getattr(result, "metrics", {}) or {}
#         metrics_dict = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
#         safe_metrics = {}
#         for key, value in metrics_dict.items():
#             try:
#                 json.dumps(value)
#                 safe_metrics[key] = value
#             except (TypeError, ValueError):
#                 safe_metrics[key] = repr(value)
#         summary = {
#             "train_path": configs["train_path"],
#             "test_path": configs["test_path"],
#             "train_save": configs["train_save"],
#             "mlflow_run_name": configs.get("mlflow", {}).get("run_name"),
#             "best_dice_score": safe_metrics.get("best_dice_score", 0.0),
#             "ray_metrics": safe_metrics,
#             "test_datasets": dataset_info.get("test_datasets", []),
#             "best_config": best_config,
#         }
#         with open(training_summary_artifact.path, "w", encoding="utf-8") as fp:
#             json.dump(summary, fp, indent=2)
#         logger.info("Training summary written to %s", training_summary_artifact.path)
#         model_artifact.metadata["train_save"] = configs["train_save"]
#     finally:
#         ray.shutdown()


# @dsl.component(base_image=BASE_IMAGE)
# def evaluate_model_component(
#     evaluation_metrics: dsl.Output[Metrics],
#     training_summary_artifact: dsl.Input[dsl.Artifact],
#     evaluation_threshold: float = 0.6,
# ) -> None:
#     """Evaluate the trained model against a simple quality bar."""
#     import json
#     import logging

#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("evaluate")

#     from pathlib import Path

#     summary = json.loads(
#         Path(training_summary_artifact.path).read_text(encoding="utf-8")
#     )
#     dice_score = float(summary.get("best_dice_score", 0.0))
#     passed = float(dice_score >= evaluation_threshold)

#     evaluation_metrics.log_metric("best_dice_score", dice_score)
#     evaluation_metrics.log_metric("passed_quality_check", passed)

#     logger.info(
#         "Evaluation result: dice=%.4f threshold=%.4f passed=%s",
#         dice_score,
#         evaluation_threshold,
#         bool(passed),
#     )

#     if dice_score < evaluation_threshold:
#         raise RuntimeError(
#             f"Quality threshold not met: dice={dice_score:.4f} < {evaluation_threshold:.4f}"
#         )


# @dsl.pipeline(name="ray-segmentation-training-pipeline")
# def ray_segmentation_training_pipeline(
#     kube_config_path: str = "training/configs/kube_configs.yaml",
#     profile: str = "",
#     env_path: str = ".env",
#     minio_endpoint: str = "",
#     minio_access_key: str = "",
#     minio_secret_key: str = "",
#     minio_secure: bool = False,
#     ray_address: str = "",
#     ray_namespace: str = "",
#     ray_storage_path: str = "",
#     mlflow_tracking_uri: str = "",
#     mlflow_s3_endpoint: str = "",
#     evaluation_threshold: float = 0.6,
# ):
#     download_task = download_dataset(
#         kube_config_path=kube_config_path,
#         env_path=env_path,
#         minio_endpoint=minio_endpoint,
#         minio_access_key=minio_access_key,
#         minio_secret_key=minio_secret_key,
#         minio_secure=minio_secure,
#     )

#     tune_task = ray_tune_component(
#         dataset_artifact=download_task.outputs["dataset_artifact"],
#         dataset_info_artifact=download_task.outputs["dataset_info_artifact"],
#         kube_config_path=kube_config_path,
#         profile=profile,
#         env_path=env_path,
#         ray_address=ray_address,
#         ray_namespace=ray_namespace,
#         ray_storage_path=ray_storage_path,
#         mlflow_tracking_uri=mlflow_tracking_uri,
#         mlflow_s3_endpoint=mlflow_s3_endpoint,
#     )

#     train_task = ray_train_component(
#         dataset_artifact=download_task.outputs["dataset_artifact"],
#         dataset_info_artifact=download_task.outputs["dataset_info_artifact"],
#         best_config_artifact=tune_task.outputs["best_config_artifact"],
#         kube_config_path=kube_config_path,
#         profile=profile,
#         env_path=env_path,
#         ray_address=ray_address,
#         ray_namespace=ray_namespace,
#         ray_storage_path=ray_storage_path,
#         mlflow_tracking_uri=mlflow_tracking_uri,
#         mlflow_s3_endpoint=mlflow_s3_endpoint,
#     )

#     evaluate_model_component(
#         training_summary_artifact=train_task.outputs["training_summary_artifact"],
#         evaluation_threshold=evaluation_threshold,
#     )


# def compile_pipeline(output_path: str) -> None:
#     compiler.Compiler().compile(
#         pipeline_func=ray_segmentation_training_pipeline,
#         package_path=output_path,
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compile the Ray segmentation KFP pipeline")
#     parser.add_argument(
#         "--output",
#         type=str,
#         default="ray_segmentation_pipeline_v3.yaml",
#         help="Destination path for the compiled pipeline YAML.",
#     )
#     args = parser.parse_args()
#     compile_pipeline(args.output)
#     print(f"Pipeline YAML written to {args.output}")

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from kfp import compiler, dsl
from kfp.dsl import Metrics, Model

BASE_IMAGE = "harly1506/polyp-mlops:kfpv2"
@dsl.component(base_image=BASE_IMAGE)
def download_dataset(
    dataset_artifact: dsl.Output[dsl.Dataset],
    dataset_info_artifact: dsl.Output[dsl.Artifact],
    kube_config_path: str = "training/configs/kube_configs.yaml",
    env_path: str = ".env",
    minio_endpoint: str = "",
    minio_access_key: str = "",
    minio_secret_key: str = "",
    minio_secure: bool = False,
) -> None:
    """Download the training and testing dataset from MinIO."""
    import logging
    import platform
    import sys
    from pathlib import Path as _Path
    import os

    import boto3
    from botocore.client import Config as BotoConfig
    from dotenv import dotenv_values
    import yaml
    import json
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("download_dataset")

    logger.info(
        "Environment: python=%s platform=%s",
        sys.version.replace("\n", " "),
        platform.platform(),
    )

    def _load_yaml_local(path: str) -> Dict[str, Any]:
        from pathlib import Path as __Path

        with __Path(path).open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}

    kube_cfg = _load_yaml_local(kube_config_path)
    bucket = kube_cfg.get("bucket")
    prefix = kube_cfg.get("dataset_prefix", "").rstrip("/")
    mlflow_cfg = kube_cfg.get("mlflow", {})
    if not bucket or not prefix:
        raise ValueError("bucket and dataset_prefix must be provided in kube_configs.yaml")

    dataset_root = _Path(dataset_artifact.path)
    dataset_root.mkdir(parents=True, exist_ok=True)

    env_values = {}
    env_file = _Path(env_path)
    if env_file.exists():
        env_values = {k: str(v) for k, v in dotenv_values(env_file).items() if v is not None}
        for key, value in env_values.items():
            os.environ.setdefault(key, value)
        logger.info("Loaded %d entries from %s", len(env_values), env_file)
    else:
        logger.warning("Env file %s not found; relying on provided parameters", env_file)

    access_key = (
        minio_access_key
        or env_values.get("AWS_ACCESS_KEY_ID")
        or env_values.get("MINIO_ROOT_USER")
    )
    secret_key = (
        minio_secret_key
        or env_values.get("AWS_SECRET_ACCESS_KEY")
        or env_values.get("MINIO_ROOT_PASSWORD")
    )
    endpoint = (
        minio_endpoint
        or env_values.get("MINIO_ENDPOINT")
        or mlflow_cfg.get("s3_endpoint_url")
        or mlflow_cfg.get("s3_endpoint")
    )

    if endpoint and not endpoint.startswith("http"):
        scheme = "https" if minio_secure else "http"
        endpoint = f"{scheme}://{endpoint.lstrip('/') }"

    if not endpoint:
        raise ValueError("MinIO endpoint must be supplied either via parameters or .env file")
    if not access_key or not secret_key:
        raise ValueError("MinIO credentials are required")

    session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    s3 = session.resource(
        "s3",
        endpoint_url=endpoint,
        config=BotoConfig(signature_version="s3v4"),
        region_name=session.region_name or "us-east-1",
    )
    bucket_obj = s3.Bucket(bucket)

    def _download_prefix(source_prefix: str, target_root: _Path) -> int:
        files_downloaded = 0
        logger.info("Downloading s3://%s/%s -> %s", bucket, source_prefix, target_root)
        for obj in bucket_obj.objects.filter(Prefix=source_prefix):
            key = obj.key
            if key.endswith("/"):
                continue
            relative_key = key[len(source_prefix) :].lstrip("/")
            destination = target_root / relative_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            bucket_obj.download_file(key, str(destination))
            files_downloaded += 1
        logger.info("Downloaded %d files from prefix %s", files_downloaded, source_prefix)
        return files_downloaded

    train_candidates = [
        f"{prefix}/TrainDataset",
        f"{prefix}/Traindataset",
    ]
    test_prefix = f"{prefix}/TestDataset"

    downloaded = 0
    train_target = dataset_root / "TrainDataset"
    for candidate in train_candidates:
        downloaded += _download_prefix(candidate, train_target)
        if downloaded:
            break
    if not downloaded:
        raise FileNotFoundError("No training files found under any TrainDataset prefix")

    test_target = dataset_root / "TestDataset"
    test_downloaded = _download_prefix(test_prefix, test_target)
    if not test_downloaded:
        logger.warning("No testing files found under prefix %s", test_prefix)

    expected_train_dirs = ["images", "masks"]
    missing_train_dirs = [
        subdir for subdir in expected_train_dirs if not (train_target / subdir).exists()
    ]
    if missing_train_dirs:
        raise FileNotFoundError(
            f"Missing expected training sub-directories under {train_target}: {missing_train_dirs}"
        )

    test_datasets = []
    if test_target.exists():
        for dataset_dir in sorted(test_target.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            required = [dataset_dir / "images", dataset_dir / "masks"]
            if not all(path.exists() for path in required):
                logger.warning(
                    "Skipping dataset %s due to missing images/masks directories", dataset_name
                )
                continue
            test_datasets.append(dataset_name)
    if not test_datasets:
        logger.warning("No test datasets discovered beneath %s", test_target)
    else:
        logger.info("Discovered %d test datasets: %s", len(test_datasets), test_datasets)

    def _summarise_directory(root: _Path, depth: int = 1) -> None:
        """Log a human readable snapshot of a directory tree."""

        def _relative(path: _Path) -> str:
            try:
                return str(path.relative_to(root)) or "."
            except ValueError:
                return str(path)

        logger.info("Directory snapshot for %s (depth=%d)", root, depth)
        queue = [(root, 0)]
        while queue:
            current, level = queue.pop(0)
            if level > depth:
                continue
            try:
                entries = sorted(current.iterdir())
            except FileNotFoundError:
                logger.warning("Missing directory during snapshot: %s", current)
                continue
            logger.info("[%s] contains %d entries", _relative(current), len(entries))
            for entry in entries:
                logger.info("  %s %s", "-" * (level + 1), _relative(entry))
                if entry.is_dir():
                    queue.append((entry, level + 1))

    _summarise_directory(dataset_root, depth=2)

    train_image_files = list((train_target / "images").rglob("*.png"))
    train_mask_files = list((train_target / "masks").rglob("*.png"))
    logger.info(
        "Training data counts: images=%d masks=%d", len(train_image_files), len(train_mask_files)
    )

    if test_target.exists():
        for dataset_name in test_datasets:
            image_dir = test_target / dataset_name / "images"
            mask_dir = test_target / dataset_name / "masks"
            image_count = len(list(image_dir.rglob("*.png")))
            mask_count = len(list(mask_dir.rglob("*.png")))
            logger.info(
                "Test dataset '%s': images=%d masks=%d", dataset_name, image_count, mask_count
            )

    info = {
        "bucket": bucket,
        "prefix": prefix,
        "root_dir": str(dataset_root),
        "train_path": str((dataset_root / "TrainDataset").resolve()),
        "test_path": str((dataset_root / "TestDataset").resolve()),
        "test_datasets": test_datasets,
    }
    with open(dataset_info_artifact.path, "w", encoding="utf-8") as fp:
        json.dump(info, fp, indent=2)
    logger.info("Dataset metadata written to %s", dataset_info_artifact.path)


@dsl.component(base_image=BASE_IMAGE,     packages_to_install=[
        "ray[client]==2.48.0",   # khớp version head
        "grpcio",                # đảm bảo có gRPC
        "protobuf<6",            # chỉ cần nếu gặp lỗi tương thích
    ],)
def ray_tune_component(
    dataset_artifact: dsl.Input[dsl.Dataset],
    dataset_info_artifact: dsl.Input[dsl.Artifact],
    best_config_artifact: dsl.Output[dsl.Artifact],
    kube_config_path: str = "training/configs/kube_configs.yaml",
    profile: str = "",
    env_path: str = ".env",
    ray_address: str = "",
    ray_namespace: str = "",
    ray_storage_path: str = "",
    mlflow_tracking_uri: str = "",
    mlflow_s3_endpoint: str = "",
) -> None:
    """Run Ray Tune to obtain the best configuration."""
    import logging
    import platform
    import sys
    import json
    import os
    from pathlib import Path as _Path

    import ray
    import torch
    from dotenv import dotenv_values

    from training import ray_main
    from training.configs.load_configs import load_config

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ray_tune")

    logger.info(
        "Environment: python=%s torch=%s ray=%s platform=%s",
        sys.version.replace("\n", " "),
        torch.__version__,
        ray.__version__,
        platform.platform(),
    )

    env_file = _Path(env_path)
    if env_file.exists():
        env_values = {k: str(v) for k, v in dotenv_values(env_file).items() if v is not None}
        for key, value in env_values.items():
            os.environ.setdefault(key, value)
        logger.info("Loaded %d env vars from %s", len(env_values), env_file)

    dataset_info = json.loads(_Path(dataset_info_artifact.path).read_text(encoding="utf-8"))
    logger.info("Resolved dataset info: %s", json.dumps(dataset_info, indent=2))

    configs = load_config(kube_config_path)
    if profile:
        configs["profile"] = profile

    configs["train_path"] = dataset_info["train_path"]
    configs["test_path"] = dataset_info["test_path"]
    dataset_root = _Path(dataset_artifact.path)
    tune_save_dir = (dataset_root / "artifacts").resolve()
    tune_save_dir.mkdir(parents=True, exist_ok=True)
    configs["train_save"] = str(tune_save_dir)

    def _merge_mlflow_config_local(
        configs: Dict[str, Any], tracking_uri: str, s3_endpoint: str
    ) -> None:
        mlflow_cfg = configs.setdefault("mlflow", {})
        if "s3_endpoint_url" in mlflow_cfg and "s3_endpoint" not in mlflow_cfg:
            mlflow_cfg["s3_endpoint"] = mlflow_cfg.get("s3_endpoint_url")
        if tracking_uri:
            mlflow_cfg["tracking_uri"] = tracking_uri
        if s3_endpoint:
            mlflow_cfg["s3_endpoint"] = s3_endpoint

    _merge_mlflow_config_local(configs, mlflow_tracking_uri, mlflow_s3_endpoint)

    mlflow_cfg = configs.setdefault("mlflow", {})
    run_name = mlflow_cfg.get("run_name")
    if not run_name or run_name == "auto":
        run_name = ray_main.create_run_name(configs)
        mlflow_cfg["run_name"] = run_name
    configs["mlflow_run_name"] = run_name

    ray_configs = dict(configs.get("ray") or {})
    if ray_address:
        ray_configs["address"] = ray_address
    if ray_namespace:
        ray_configs["namespace"] = ray_namespace
    if ray_storage_path:
        ray_configs["storage_path"] = ray_storage_path
    configs["ray"] = ray_configs

    ray_main.setup_logging(str(configs.get("log_level", "INFO")))
    ray_main.resolve_mlflow_env(configs)
    ray_main.seed_everything(int(configs.get("seed", 1234)))

    ray_main._override_ray_from_env(ray_configs)
    storage_path = ray_main.resolve_storage_path(configs)
    scaling_config = ray_main.build_scaling_config(configs)

    init_kwargs = {
        "address": ray_configs.get("address"),
        "namespace": ray_configs.get("namespace"),
        "runtime_env": ray_configs.get("runtime_env"),
        "ignore_reinit_error": True,
        "local_mode": bool(ray_configs.get("local_mode", False)),
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v}
    
    logger.info("Final Ray init kwargs: %s", json.dumps(init_kwargs, indent=2))
    logger.info("Resolved Ray storage path: %s", storage_path)
    logger.info("Resolved Ray scaling config: %s", scaling_config)
    logger.info("Resolved training save dir for tuning: %s", configs["train_save"])
    logger.info("Resolved MLflow config: %s", json.dumps(configs.get("mlflow", {}), indent=2))

    logger.info("Dataset artifact directory listing:")
    for path in sorted(_Path(dataset_artifact.path).glob("*")):
        logger.info(" - %s", path)

    ray.init(**init_kwargs)

    try:
        best_config = ray_main.run_tuning(configs, scaling_config, storage_path)
        logger.info("Best hyperparameters: %s", json.dumps(best_config, indent=2))
        with open(best_config_artifact.path, "w", encoding="utf-8") as fp:
            json.dump(best_config, fp, indent=2)
    finally:
        ray.shutdown()


@dsl.component(base_image=BASE_IMAGE,     packages_to_install=[
        "ray[client]==2.48.0",   # khớp version head
        "grpcio",                # đảm bảo có gRPC
        "protobuf<6",            # chỉ cần nếu gặp lỗi tương thích
    ],)
def ray_train_component(
    dataset_artifact: dsl.Input[dsl.Dataset],
    dataset_info_artifact: dsl.Input[dsl.Artifact],
    best_config_artifact: dsl.Input[dsl.Artifact],
    model_artifact: dsl.Output[Model],
    training_summary_artifact: dsl.Output[dsl.Artifact],
    kube_config_path: str = "training/configs/kube_configs.yaml",
    profile: str = "",
    env_path: str = ".env",
    ray_address: str = "",
    ray_namespace: str = "",
    ray_storage_path: str = "",
    mlflow_tracking_uri: str = "",
    mlflow_s3_endpoint: str = "",
) -> None:
    """Run the final Ray training job using the best configuration."""
    import logging
    import platform
    import sys
    import json
    import os
    from pathlib import Path as _Path

    import ray
    import torch
    from dotenv import dotenv_values

    from training import ray_main
    from training.configs.load_configs import load_config

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ray_train")

    logger.info(
        "Environment: python=%s torch=%s ray=%s platform=%s",
        sys.version.replace("\n", " "),
        torch.__version__,
        ray.__version__,
        platform.platform(),
    )

    env_file = _Path(env_path)
    if env_file.exists():
        env_values = {k: str(v) for k, v in dotenv_values(env_file).items() if v is not None}
        for key, value in env_values.items():
            os.environ.setdefault(key, value)
        logger.info("Loaded %d env vars from %s", len(env_values), env_file)

    dataset_info = json.loads(_Path(dataset_info_artifact.path).read_text(encoding="utf-8"))
    logger.info("Resolved dataset info: %s", json.dumps(dataset_info, indent=2))
    best_config = json.loads(_Path(best_config_artifact.path).read_text(encoding="utf-8"))
    logger.info("Loaded best config overrides: %s", json.dumps(best_config, indent=2))

    configs = load_config(kube_config_path)
    if profile:
        configs["profile"] = profile

    configs["train_path"] = dataset_info["train_path"]
    configs["test_path"] = dataset_info["test_path"]
    dataset_root = _Path(dataset_artifact.path)
    model_dir = _Path(model_artifact.path).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    configs["train_save"] = str(model_dir)

    def _merge_mlflow_config_local(
        configs: Dict[str, Any], tracking_uri: str, s3_endpoint: str
    ) -> None:
        mlflow_cfg = configs.setdefault("mlflow", {})
        if "s3_endpoint_url" in mlflow_cfg and "s3_endpoint" not in mlflow_cfg:
            mlflow_cfg["s3_endpoint"] = mlflow_cfg.get("s3_endpoint_url")
        if tracking_uri:
            mlflow_cfg["tracking_uri"] = tracking_uri
        if s3_endpoint:
            mlflow_cfg["s3_endpoint"] = s3_endpoint

    _merge_mlflow_config_local(configs, mlflow_tracking_uri, mlflow_s3_endpoint)

    mlflow_cfg = configs.setdefault("mlflow", {})
    run_name = mlflow_cfg.get("run_name")
    if not run_name or run_name == "auto":
        run_name = ray_main.create_run_name(configs)
    final_run_name = f"{run_name}_final"
    mlflow_cfg["run_name"] = final_run_name
    configs["mlflow_run_name"] = final_run_name
    configs["is_final_training"] = True

    ray_main._deep_update(configs, best_config)

    ray_configs = dict(configs.get("ray") or {})
    if ray_address:
        ray_configs["address"] = ray_address
    if ray_namespace:
        ray_configs["namespace"] = ray_namespace
    if ray_storage_path:
        ray_configs["storage_path"] = ray_storage_path
    configs["ray"] = ray_configs

    ray_main.setup_logging(str(configs.get("log_level", "INFO")))
    ray_main.resolve_mlflow_env(configs)
    ray_main.seed_everything(int(configs.get("seed", 1234)))

    ray_main._override_ray_from_env(ray_configs)
    storage_path = ray_main.resolve_storage_path(configs)
    scaling_config = ray_main.build_scaling_config(configs)

    init_kwargs = {
        "address": ray_configs.get("address"),
        "namespace": ray_configs.get("namespace"),
        "runtime_env": ray_configs.get("runtime_env"),
        "ignore_reinit_error": True,
        "local_mode": bool(ray_configs.get("local_mode", False)),
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v}

    logger.info("Final Ray init kwargs: %s", json.dumps(init_kwargs, indent=2))
    logger.info("Resolved Ray storage path: %s", storage_path)
    logger.info("Resolved Ray scaling config: %s", json.dumps(scaling_config, indent=2))
    logger.info("Resolved model output directory: %s", configs["train_save"])
    logger.info("Resolved MLflow config: %s", json.dumps(configs.get("mlflow", {}), indent=2))

    logger.info("Dataset artifact directory listing:")
    for path in sorted(_Path(dataset_artifact.path).glob("*")):
        logger.info(" - %s", path)

    ray.init(**init_kwargs)

    try:
        result = ray_main.run_final_training(configs, scaling_config, storage_path)
        raw_metrics = getattr(result, "metrics", {}) or {}
        metrics_dict = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
        safe_metrics = {}
        for key, value in metrics_dict.items():
            try:
                json.dumps(value)
                safe_metrics[key] = value
            except (TypeError, ValueError):
                safe_metrics[key] = repr(value)
        summary = {
            "train_path": configs["train_path"],
            "test_path": configs["test_path"],
            "train_save": configs["train_save"],
            "mlflow_run_name": configs.get("mlflow", {}).get("run_name"),
            "best_dice_score": safe_metrics.get("best_dice_score", 0.0),
            "ray_metrics": safe_metrics,
            "test_datasets": dataset_info.get("test_datasets", []),
            "best_config": best_config,
        }
        with open(training_summary_artifact.path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        logger.info("Training summary written to %s", training_summary_artifact.path)
        model_artifact.metadata["train_save"] = configs["train_save"]
        logger.info(
            "Ray result metrics (%d keys): %s",
            len(safe_metrics),
            json.dumps(safe_metrics, indent=2),
        )
    finally:
        ray.shutdown()


@dsl.component(base_image=BASE_IMAGE)
def evaluate_model_component(
    evaluation_metrics: dsl.Output[Metrics],
    training_summary_artifact: dsl.Input[dsl.Artifact],
    evaluation_threshold: float = 0.6,
) -> None:
    """Evaluate the trained model against a simple quality bar."""
    import json
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate")

    from pathlib import Path

    summary_path = Path(training_summary_artifact.path)
    logger.info("Reading training summary from %s", summary_path)
    summary_text = summary_path.read_text(encoding="utf-8")
    logger.info("Training summary payload: %s", summary_text)
    summary = json.loads(summary_text)
    dice_score = float(summary.get("best_dice_score", 0.0))
    passed = float(dice_score >= evaluation_threshold)

    evaluation_metrics.log_metric("best_dice_score", dice_score)
    evaluation_metrics.log_metric("passed_quality_check", passed)

    logger.info(
        "Evaluation result: dice=%.4f threshold=%.4f passed=%s",
        dice_score,
        evaluation_threshold,
        bool(passed),
    )

    if dice_score < evaluation_threshold:
        raise RuntimeError(
            f"Quality threshold not met: dice={dice_score:.4f} < {evaluation_threshold:.4f}"
        )


@dsl.pipeline(name="ray-segmentation-training-pipeline")
def ray_segmentation_training_pipeline(
    kube_config_path: str = "training/configs/kube_configs.yaml",
    profile: str = "",
    env_path: str = ".env",
    minio_endpoint: str = "",
    minio_access_key: str = "",
    minio_secret_key: str = "",
    minio_secure: bool = False,
    ray_address: str = "",
    ray_namespace: str = "",
    ray_storage_path: str = "",
    mlflow_tracking_uri: str = "",
    mlflow_s3_endpoint: str = "",
    evaluation_threshold: float = 0.6,
):
    download_task = download_dataset(
        kube_config_path=kube_config_path,
        env_path=env_path,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_secure=minio_secure,
    )

    tune_task = ray_tune_component(
        dataset_artifact=download_task.outputs["dataset_artifact"],
        dataset_info_artifact=download_task.outputs["dataset_info_artifact"],
        kube_config_path=kube_config_path,
        profile=profile,
        env_path=env_path,
        ray_address=ray_address,
        ray_namespace=ray_namespace,
        ray_storage_path=ray_storage_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint=mlflow_s3_endpoint,
    )

    train_task = ray_train_component(
        dataset_artifact=download_task.outputs["dataset_artifact"],
        dataset_info_artifact=download_task.outputs["dataset_info_artifact"],
        best_config_artifact=tune_task.outputs["best_config_artifact"],
        kube_config_path=kube_config_path,
        profile=profile,
        env_path=env_path,
        ray_address=ray_address,
        ray_namespace=ray_namespace,
        ray_storage_path=ray_storage_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint=mlflow_s3_endpoint,
    )

    evaluate_model_component(
        training_summary_artifact=train_task.outputs["training_summary_artifact"],
        evaluation_threshold=evaluation_threshold,
    )


def compile_pipeline(output_path: str) -> None:
    compiler.Compiler().compile(
        pipeline_func=ray_segmentation_training_pipeline,
        package_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile the Ray segmentation KFP pipeline")
    parser.add_argument(
        "--output",
        type=str,
        default="ray_segmentation_pipeline_v4.yaml",
        help="Destination path for the compiled pipeline YAML.",
    )
    args = parser.parse_args()
    compile_pipeline(args.output)
    print(f"Pipeline YAML written to {args.output}")