import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple  
from urllib.parse import urlparse

import numpy as np
import torch
from ray.train import RunConfig, ScalingConfig, get_context
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model
from ray.tune.tuner import Tuner

import ray
from ray import tune
from training.configs.load_configs import load_config
from training.models.unet import UNet
from training.trainer.ray_trainer import SegmentationTrainer
from training.utils.dataset import get_loader

# boto3 guard — để tránh NameError nếu môi trường worker chưa có boto3
try:
    import boto3
    from botocore.client import Config as BotoConfig
except Exception:  # noqa: BLE001
    boto3 = None
    BotoConfig = None

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def resolve_mlflow_env(configs: Dict[str, Any]) -> None:
    """
    Resolve and set MLflow environment variables based on the provided configs.
    Make sure we have MLFLOW_TRACKING_URI and optionally MLFLOW_S3_ENDPOINT_URL set in env. or configs.
    """
    mlflow_cfg: Dict[str, Any] = configs.get("mlflow", {})
    if not mlflow_cfg.get("enable"):
        return

    uri = mlflow_cfg.get("tracking_uri") or os.environ.get(
        mlflow_cfg.get("tracking_uri", "MLFLOW_TRACKING_URI")
    )
    if uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", str(uri))

    s3_endpoint = mlflow_cfg.get("s3_endpoint") or os.environ.get(
        mlflow_cfg.get("s3_endpoint_env", "MLFLOW_S3_ENDPOINT_URL")
    )
    if s3_endpoint:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", str(s3_endpoint))


def create_model(model_name: str):
    # For simplicity, we only implement UNet here. Extend as needed.
    if model_name.lower() != "unet":
        logging.warning("Model %s not recognised, defaulting to UNet", model_name)
    return UNet(1)


def create_run_name(configs: Dict[str, Any]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return (
        f"{configs.get('model', 'model')}_bs{configs.get('batchsize', '')}"
        f"_lr{configs.get('lr', '')}_ep{configs.get('epoch', '')}_{timestamp}"
    )


def build_scaling_config(configs: Dict[str, Any]) -> ScalingConfig:
    scaling = (configs.get("ray") or {}).get("scaling", {})
    use_gpu_cfg = scaling.get("use_gpu", "auto")
    use_gpu = (use_gpu_cfg is True) or (
        use_gpu_cfg == "auto" and torch.cuda.is_available()
    )

    resources_per_worker = dict(scaling.get("resources_per_worker", {"CPU": 1}))
    if use_gpu:
        resources_per_worker.setdefault("GPU", 1)

    return ScalingConfig(
        num_workers=int(scaling.get("num_workers", 1)),
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
    )


def _split_s3_uri(uri: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (bucket, key) from s3://bucket/key...; if not s3, return (None, None)."""
    if not uri or not uri.startswith("s3://"):
        return None, None
    rest = uri[5:]  # strip 's3://'
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _normalise_endpoint(endpoint: Optional[str], secure: Optional[bool]) -> Optional[str]:
    if not endpoint:
        return None
    endpoint = str(endpoint)
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    scheme = "https" if secure else "http"
    return f"{scheme}://{endpoint.lstrip('/')}"


def _resolve_worker_root(config: Dict[str, Any]) -> Path:
    root = config.get("worker_dataset_root") or os.getenv(
        "RAY_WORKER_DATASET_ROOT", "/tmp/ray-dataset"
    )
    return Path(root).expanduser().resolve()


def _ensure_train_save_dir(config: Dict[str, Any], worker_root: Path) -> str:
    desired = config.get("train_save")
    candidate = Path(desired).expanduser() if desired else worker_root / "artifacts"
    if not candidate.is_absolute():
        candidate = (worker_root / candidate).resolve()
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        candidate = worker_root / "artifacts"
        candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def _download_dataset_from_minio(
    worker_root: Path, config: Dict[str, Any], dataset_info: Optional[Dict[str, Any]] = None
) -> Tuple[str, Optional[str]]:
    """
    Materialise S3 dataset (MinIO) to worker_root/TrainDataset (+ TestDataset if any).
    Priority to get bucket/prefix from config['train_path'] as s3://..., fallback to dataset_info.
    Return (local_train_dir, local_test_dir_or_None)
    """
    if boto3 is None or BotoConfig is None:
        raise RuntimeError("boto3 is required on Ray workers to download dataset from MinIO/S3")

    dataset_info = dataset_info or {}

    # bucket + train_prefix + test_prefix from config or dataset_info
    b_cfg, train_key_cfg = _split_s3_uri(config.get("train_path"))
    b_info, train_key_info = _split_s3_uri(dataset_info.get("train_path"))
    bucket = b_cfg or dataset_info.get("bucket") or b_info
    train_prefix = train_key_cfg or dataset_info.get("train_source_prefix") or dataset_info.get("train_prefix") or train_key_info
    _, test_key_cfg = _split_s3_uri(config.get("test_path"))
    _, test_key_info = _split_s3_uri(dataset_info.get("test_path"))
    test_prefix = test_key_cfg or dataset_info.get("test_source_prefix") or test_key_info

    if not bucket or not train_prefix:
        raise RuntimeError("Cannot resolve S3 dataset location (bucket/train_prefix) from config/dataset_info")

    #end point end credentials for boto3, first from env, then from dataset_info
    endpoint = (
        os.getenv("AWS_ENDPOINT_URL")
        or dataset_info.get("s3_endpoint")
        or os.getenv("MLFLOW_S3_ENDPOINT_URL")
        or dataset_info.get("minio_endpoint_used")
    )
    #if endpoint do not have scheme, add http or https based on minio_secure (default http)
    endpoint = _normalise_endpoint(endpoint, dataset_info.get("minio_secure"))

    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD")

    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("Missing MinIO/S3 endpoint or credentials on Ray worker")

    session = boto3.session.Session(
        aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )
    s3 = session.resource(
        "s3",
        endpoint_url=endpoint,
        config=BotoConfig(signature_version="s3v4"),
        region_name=session.region_name or "us-east-1",
    )
    bucket_obj = s3.Bucket(bucket)

    logger = logging.getLogger("dataset")
    worker_root.mkdir(parents=True, exist_ok=True)

    def _download_prefix(prefix: str, target_dir: Path) -> int:
        prefix = prefix.rstrip("/")
        count = 0
        for obj in bucket_obj.objects.filter(Prefix=prefix):
            key = obj.key
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/")
            dst = target_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            bucket_obj.download_file(key, str(dst))
            count += 1
        logger.info("Downloaded %d files from s3://%s/%s -> %s", count, bucket, prefix, target_dir)
        return count

    # Download train and test (if any)
    train_dir = (worker_root / "TrainDataset").resolve()
    _download_prefix(train_prefix, train_dir)

    test_dir = None
    if test_prefix:
        test_dir = (worker_root / "TestDataset").resolve()
        _download_prefix(test_prefix, test_dir)

    return str(train_dir), (str(test_dir) if test_dir else None)


def _ensure_dataset_available(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (local_train_path, local_test_path) for local use.
    If s3://... download to worker root and return local paths.
    """
    logger = logging.getLogger("dataset")
    dataset_info = config.get("dataset_info") or {}  # optional
    worker_root = _resolve_worker_root(config)

    #check if local path then we can use it directly (for dev)
    train_path_cfg = config.get("train_path") or ""
    if not str(train_path_cfg).startswith("s3://"):
        train_path = Path(train_path_cfg).expanduser()
        test_path = Path(config.get("test_path", train_path.parent / "TestDataset")).expanduser()
        if (train_path / "images").is_dir() and (train_path / "masks").is_dir():
            return str(train_path), str(test_path)

    #if train_path is s3://... then we need to download it to worker local
    logger.info("Materialising dataset from S3 to worker-local storage...")
    local_train, local_test = _download_dataset_from_minio(worker_root, config, dataset_info)
    # make sure we have images/ and masks/ in train dir
    if not (Path(local_train) / "images").is_dir() or not (Path(local_train) / "masks").is_dir():
        raise AssertionError(f"Missing train dirs after download: {local_train}/images / {local_train}/masks")
    return local_train, (local_test or str(Path(local_train).parent / "TestDataset"))


def train_func(config: Dict[str, Any]):

    cfg = dict(config)  
    ctx = get_context()
    rank = int(ctx.get_world_rank())
    logger = logging.getLogger(f"worker_{rank}")

    try:
        # make sure dataset and train_save are ready
        orig_train = cfg.get("train_path")
        orig_test = cfg.get("test_path")
        train_path, test_path = _ensure_dataset_available(cfg)
        cfg["train_path"] = train_path
        cfg["test_path"] = test_path

        if train_path != orig_train:
            logger.info("Using worker-local train dir: %s (was %s)", train_path, orig_train)
        if test_path and test_path != orig_test:
            logger.info("Using worker-local test dir: %s (was %s)", test_path, orig_test)

        worker_root = _resolve_worker_root(cfg)
        cfg["train_save"] = _ensure_train_save_dir(cfg, worker_root)
        logger.info("Training artifacts will be stored under: %s", cfg["train_save"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images_dir = os.path.join(train_path, "images")
        masks_dir = os.path.join(train_path, "masks")
        assert os.path.isdir(images_dir) and os.path.isdir(masks_dir), \
            f"Missing train dirs: {images_dir} / {masks_dir}"

        train_loader = get_loader(
            images_dir,
            masks_dir,
            int(cfg["batchsize"]),
            int(cfg["trainsize"]),
            bool(cfg.get("augmentation", False)),
        )
        train_loader = prepare_data_loader(train_loader)

        model = create_model(cfg["model"])
        model = prepare_model(model).to(device)

        trainer = SegmentationTrainer(
            model=model,
            train_loader=train_loader,
            test_path=cfg["test_path"],
            configs=cfg,
            world_rank=rank,
        )

        trainer.train()

    except Exception:
        logger.exception("Training failed")
        raise
# def train_func(config: Dict[str, Any]):
#     ctx = get_context()
#     rank = int(ctx.get_world_rank())
#     logger = logging.getLogger(f"worker_{rank}")

#     try:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         images_dir = os.path.join(config["train_path"], "images")
#         masks_dir = os.path.join(config["train_path"], "masks")
#         assert os.path.isdir(images_dir) and os.path.isdir(
#             masks_dir
#         ), f"Missing train dirs: {images_dir} / {masks_dir}"

#         train_loader = get_loader(
#             images_dir,
#             masks_dir,
#             int(config["batchsize"]),
#             int(config["trainsize"]),
#             bool(config.get("augmentation", False)),
#         )
#         train_loader = prepare_data_loader(train_loader)

#         model = create_model(config["model"])
#         model = prepare_model(model).to(device)

#         trainer = SegmentationTrainer(
#             model=model,
#             train_loader=train_loader,
#             test_path=config["test_path"],
#             configs=config,
#             world_rank=rank,
#         )

#         # best_dice = 0.0
#         # for epoch in range(1, int(config["epoch"]) + 1):
#         #     if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
#         #         train_loader.sampler.set_epoch(int(epoch))

#         #     trainer.train_epoch(int(epoch))

#         #     current_dice = 0.0
#         #     if rank == 0:
#         #         trainer.validate(int(epoch))
#         #         current_dice = float(trainer.dice_scores.get("test", [0.0])[-1])
#         #         trainer.scheduler.step(current_dice)
#         #         best_dice = max(best_dice, current_dice)

#         #         ray.train.report({
#         #             "epoch": int(epoch),
#         #             "dice": float(current_dice),
#         #             "best_dice": float(best_dice),
#         #         })
#         #     else:
#         #         ray.train.report({
#         #             "epoch": int(epoch),
#         #             "dice": 0.0,
#         #             "best_dice": float(best_dice),
#         #         })

#         trainer.train()
#     except Exception:
#         logger.exception("Training failed")
#         raise


def _deep_update(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)
        else:
            base[key] = value


def run_tuning(
    configs: Dict[str, Any], scaling_config: ScalingConfig, storage_path: str
) -> Dict[str, Any]:
    space = {
        "train_loop_config": {
            "batchsize": tune.choice(configs["tuning"]["search_space"]["batchsize"]),
            "lr": tune.choice(configs["tuning"]["search_space"]["lr"]),
            "trainsize": tune.choice(configs["tuning"]["search_space"]["trainsize"]),
            "decay_rate": tune.choice(configs["tuning"]["search_space"]["decay_rate"]),
            "clip": tune.choice(configs["tuning"]["search_space"]["clip"]),
        }
    }
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        train_loop_config=configs,
    )
    tuner = Tuner(
        trainer,
        run_config=RunConfig(
            storage_path=storage_path,
            name=f"{configs["mlflow"]["run_name"]}_tuning",
            checkpoint_config=tune.CheckpointConfig(num_to_keep=1),
            # callbacks=[MLflowLoggerCallback(tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            # experiment_name=configs.get("mlflow_experiment_name", "PolypSegmentation"),
            # save_artifact=True,
            # )],
        ),
        tune_config=tune.TuneConfig(
            metric=configs["tuning"]["metric"],
            mode=configs["tuning"]["mode"],
            num_samples=configs["tuning"]["num_samples"],
        ),
        param_space=space,
    )
    res = tuner.fit()
    return res.get_best_result(
        metric=configs["tuning"]["metric"], mode=configs["tuning"]["mode"]
    ).config["train_loop_config"]


def run_final_training(
    configs: Dict[str, Any],
    scaling_config: ScalingConfig,
    storage_path: str,
) -> Any:
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        train_loop_config=configs,
        run_config=RunConfig(
            storage_path=storage_path,
            name=configs.get("mlflow_run_name", "ray_training_run"),
        ),
    )
    return trainer.fit()


# def _maybe_patch_runtime_env(ray_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#     runtime_env = ray_cfg.get("runtime_env")
#     if not runtime_env:
#         return None
#     if not bool(ray_cfg.get("patch_runtime_env", True)):
#         return runtime_env

#     runtime_env = dict(runtime_env)
#     runtime_env.pop("pip", None)
#     runtime_env.pop("conda", None)

#     env_vars = dict(runtime_env.get("env_vars") or {})
#     env_vars.setdefault("UV_NO_PROJECT", "1")
#     env_vars.setdefault("PIP_REQUIRE_VENV", "0")
#     env_vars.setdefault("PYTHONNOUSERSITE", "1")
#     env_vars.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
#     env_vars.setdefault("VIRTUAL_ENV", "")

#     python_bin = os.path.dirname(sys.executable)
#     env_vars.setdefault("PATH", python_bin + os.pathsep + os.environ.get("PATH", ""))

#     runtime_env["env_vars"] = env_vars
#     return runtime_env


def _override_ray_from_env(ray_cfg: MutableMapping[str, Any]) -> None:
    env_map = {
        "address": os.getenv("RAY_ADDRESS"),
        "namespace": os.getenv("RAY_NAMESPACE"),
        "storage_path": os.getenv("RAY_STORAGE_PATH"),
    }
    for key, value in env_map.items():
        if value:
            ray_cfg[key] = value

    runtime_env = ray_cfg.get("runtime_env") or {}
    working_dir = os.getenv("RAY_RUNTIME_WORKING_DIR")
    if working_dir:
        runtime_env = dict(runtime_env)
        runtime_env["working_dir"] = working_dir
        ray_cfg["runtime_env"] = runtime_env


def _as_storage_uri(path: str) -> str:
    parsed = urlparse(path)
    if parsed.scheme:
        return path
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return f"file://{directory}"


def resolve_storage_path(configs: Dict[str, Any]) -> str:
    ray_cfg = configs.get("ray") or {}
    storage_path = ray_cfg.get("storage_path")
    if storage_path:
        return _as_storage_uri(str(storage_path))

    default_dir = Path("training/logs/ray").expanduser().resolve()
    default_dir.mkdir(parents=True, exist_ok=True)
    return f"file://{default_dir}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config_path", type=str, default="./training/configs/configs.yaml"
    )
    ap.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Optional profile label used only for logging; configuration overrides"
            " should be provided via the config file passed to --config_path."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_config(args.config_path)
    if args.profile:
        configs["profile"] = args.profile

    setup_logging(str(configs.get("log_level", "INFO")))

    seed_everything(int(configs.get("seed", 1234)))

    resolve_mlflow_env(configs)

    if configs.get("mlflow", {}).get("run_name") == "auto" or not configs.get(
        "mlflow", {}
    ).get("run_name"):
        configs["mlflow"]["run_name"] = create_run_name(configs)

    ray_configs: Dict[str, Any] = dict(configs.get("ray") or {})
    _override_ray_from_env(ray_configs)
    # runtime_env = _maybe_patch_runtime_env(ray_configs)
    configs["ray"] = ray_configs

    address = ray_configs.get("address")
    namespace = ray_configs.get("namespace")
    local_mode = bool(ray_configs.get("local_mode", False))

    storage_path = resolve_storage_path(configs)

    init_kwargs: Dict[str, Any] = {
        "namespace": namespace,
        "runtime_env": ray_configs.get("runtime_env"),
        "ignore_reinit_error": True,
        "local_mode": local_mode,
    }
    if address:
        init_kwargs["address"] = address

    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    logging.info("Initialising Ray with args: %s", json.dumps(init_kwargs, indent=2))

    scaling_config = build_scaling_config(configs)
    print("scaling_config:", scaling_config)
    try:
        ray.init(**init_kwargs)

        if configs.get("tuning", {}).get("enable", False):
            logging.info("Starting Ray Tune for hyper-parameter search")
            best_configs = run_tuning(configs, scaling_config, storage_path)
            _deep_update(configs, best_configs)
            logging.info("Best tuning result: %s", json.dumps(best_configs, indent=2))

        should_train = bool(configs.get("is_final_training", True))
        if should_train:
            configs["mlflow"]["run_name"] = f'{configs["mlflow"]["run_name"]}_final'
            logging.info(
                "Running final training with configs:\n%s",
                json.dumps(configs, indent=2),
            )
            result = run_final_training(configs, scaling_config, storage_path)
            logging.info("Training result: %s", result)
        else:
            logging.info("Skipping final training because is_final_training is false")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
