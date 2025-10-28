


import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from kfp import compiler, dsl
from kfp.dsl import Metrics, Model

BASE_IMAGE = "harly1506/polyp-mlops:kfpv4"
@dsl.component(base_image=BASE_IMAGE)
def inspect_dataset_component(
    dataset_artifact: dsl.Output[dsl.Dataset],
    dataset_info_artifact: dsl.Output[dsl.Artifact],
    kube_config_path: str = "training/configs/kube_configs.yaml",
    env_path: str = ".env",
    minio_endpoint: str = "",
    minio_access_key: str = "",
    minio_secret_key: str = "",
    minio_secure: bool = False,
) -> None:
    """Inspect the dataset layout in MinIO and prepare local placeholders."""
    import logging
    import platform
    import sys
    from collections import defaultdict
    from pathlib import Path as _Path
    from typing import Any, Dict
    import json
    import os
    import yaml
    import boto3
    from botocore.client import Config as BotoConfig
    from dotenv import dotenv_values

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("inspect_dataset")

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
    train_target = dataset_root / "TrainDataset"
    (train_target / "images").mkdir(parents=True, exist_ok=True)
    (train_target / "masks").mkdir(parents=True, exist_ok=True)
    test_target = dataset_root / "TestDataset"
    test_target.mkdir(parents=True, exist_ok=True)

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

    def _scan_train_prefix(source_prefix: str):
        prefix = source_prefix.rstrip("/")
        counts = {"images": 0, "masks": 0, "other": 0}
        samples = []
        for obj in bucket_obj.objects.filter(Prefix=prefix):
            key = obj.key
            if key.endswith("/"):
                continue
            relative_key = key[len(prefix) :].lstrip("/")
            if not relative_key:
                continue
            top_level = relative_key.split("/", 1)[0]
            if top_level == "images":
                counts["images"] += 1
            elif top_level == "masks":
                counts["masks"] += 1
            else:
                counts["other"] += 1
            if len(samples) < 10:
                samples.append(relative_key)
        counts["total"] = counts["images"] + counts["masks"] + counts["other"]
        return counts, samples

    def _scan_test_prefix(source_prefix: str):
        prefix = source_prefix.rstrip("/")
        dataset_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"images": 0, "masks": 0, "other": 0, "samples": []}
        )
        for obj in bucket_obj.objects.filter(Prefix=prefix):
            key = obj.key
            if key.endswith("/"):
                continue
            relative_key = key[len(prefix) :].lstrip("/")
            if not relative_key:
                continue
            parts = relative_key.split("/")
            dataset_name = parts[0] if parts else ""
            subdir = parts[1] if len(parts) > 1 else ""
            entry = dataset_stats[dataset_name]
            if subdir == "images":
                entry["images"] += 1
            elif subdir == "masks":
                entry["masks"] += 1
            else:
                entry["other"] += 1
            if len(entry["samples"]) < 5:
                entry["samples"].append(relative_key)
        return dataset_stats

    train_candidates = [
        f"{prefix}/TrainDataset",
        f"{prefix}/Traindataset",
    ]
    test_prefix = f"{prefix}/TestDataset"

    selected_train_prefix = None
    train_counts = {"images": 0, "masks": 0, "other": 0, "total": 0}
    train_samples: list[str] = []
    for candidate in train_candidates:
        counts, samples = _scan_train_prefix(candidate)
        if counts["total"]:
            selected_train_prefix = candidate.rstrip("/")
            train_counts = counts
            train_samples = samples
            break
    if not selected_train_prefix:
        raise FileNotFoundError("No training files found under any TrainDataset prefix")

    logger.info(
        "Training data overview: images=%d masks=%d other=%d total=%d",
        train_counts["images"],
        train_counts["masks"],
        train_counts["other"],
        train_counts["total"],
    )
    if train_samples:
        logger.info("Sample training keys: %s", train_samples)

    test_stats = _scan_test_prefix(test_prefix)
    test_datasets = sorted(name for name in test_stats.keys() if name)
    if not test_datasets:
        logger.warning("No test datasets discovered beneath prefix %s", test_prefix)
    else:
        logger.info("Discovered %d test datasets: %s", len(test_datasets), test_datasets)
        for dataset_name in test_datasets:
            stats = test_stats[dataset_name]
            total = stats["images"] + stats["masks"] + stats["other"]
            logger.info(
                "Test dataset '%s': images=%d masks=%d other=%d total=%d",
                dataset_name,
                stats["images"],
                stats["masks"],
                stats["other"],
                total,
            )
            if stats["samples"]:
                logger.info(
                    "  Sample keys for %s: %s", dataset_name, stats["samples"]
                )
            (test_target / dataset_name / "images").mkdir(parents=True, exist_ok=True)
            (test_target / dataset_name / "masks").mkdir(parents=True, exist_ok=True)

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

    configured_train_path = kube_cfg.get("train_path")
    if not configured_train_path:
        configured_train_path = f"s3://{bucket}/{selected_train_prefix.lstrip('/')}"

    configured_test_path = kube_cfg.get("test_path")
    if not configured_test_path:
        configured_test_path = f"s3://{bucket}/{test_prefix.lstrip('/')}"

    logger.info("Configured remote train path: %s", configured_train_path)
    logger.info("Configured remote test path: %s", configured_test_path)

    info = {
        "bucket": bucket,
        "prefix": prefix,
        "root_dir": str(dataset_root),
        "train_path": str(configured_train_path),
        "test_path": str(configured_test_path),
        "local_train_dir": str((dataset_root / "TrainDataset").resolve()),
        "local_test_dir": str((dataset_root / "TestDataset").resolve()),
        "relative_train_dir": "TrainDataset",
        "relative_test_dir": "TestDataset",
        "test_datasets": test_datasets,
        "train_prefix": f"{prefix}/TrainDataset",
        "test_prefix": test_prefix,
        "train_source_prefix": selected_train_prefix,
        "test_source_prefix": test_prefix if test_datasets else "",
        "train_files_downloaded": train_counts["total"],
        "test_files_downloaded": sum(
            stats["images"] + stats["masks"] + stats["other"]
            for stats in test_stats.values()
        ),
        "train_image_count": train_counts["images"],
        "train_mask_count": train_counts["masks"],
        "test_dataset_counts": {
            name: {
                "images": test_stats[name]["images"],
                "masks": test_stats[name]["masks"],
                "other": test_stats[name]["other"],
            }
            for name in test_datasets
        },
        "minio_endpoint_used": endpoint,
        "minio_secure": bool(minio_secure),
    }
    with open(dataset_info_artifact.path, "w", encoding="utf-8") as fp:
        json.dump(info, fp, indent=2)
    logger.info("Dataset metadata written to %s", dataset_info_artifact.path)


@dsl.component(base_image=BASE_IMAGE)
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
    config_overrides: str = "",

) -> None:
    """Run Ray Tune to obtain the best configuration."""
    import logging
    import platform
    import sys
    from pathlib import Path as _Path
    import json
    import os
    from typing import Any, Dict
    import ray
    import torch
    from dotenv import dotenv_values
    import yaml

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
    mlflow_cfg = dict(configs.get("mlflow") or {})
    if profile:
        configs["profile"] = profile

    #placeholder paths, that won't actually be used for training, just for checking paths and showing in the logs
    dataset_root = _Path(dataset_artifact.path).resolve()
    local_train_path = dataset_root / "TrainDataset"
    local_test_path = dataset_root / "TestDataset"
    logger.info("Resolved local dataset root: %s", dataset_root)
    logger.info("Resolved training directory: %s", local_train_path)
    logger.info("Resolved test directory: %s", local_test_path)
    discovered_tests = dataset_info.get("test_datasets", [])
    if discovered_tests:
        logger.info("Download step discovered test datasets: %s", discovered_tests)
    try:
        available_test_dirs = [
            p.name for p in local_test_path.iterdir() if p.is_dir()
        ]
    except FileNotFoundError:
        available_test_dirs = []
    logger.info("Test directories available locally: %s", available_test_dirs)
    if not local_train_path.exists():
        logger.warning("Training dataset not found locally at %s", local_train_path)

    remote_train_path = dataset_info.get("train_path") or configs.get("train_path")
    remote_test_path = dataset_info.get("test_path") or configs.get("test_path")
    if not remote_train_path:
        raise ValueError("Remote train_path is required in dataset metadata or config")
    if not remote_test_path:
        logger.warning("Remote test_path missing; falling back to local TestDataset path")
        remote_test_path = str(local_test_path)

    configs["train_path"] = str(remote_train_path)
    configs["test_path"] = str(remote_test_path)
    configs["local_train_dir"] = str(local_train_path)
    configs["local_test_dir"] = str(local_test_path)
    configs["dataset_info"] = dataset_info
    configs.setdefault(
        "worker_dataset_root",
        os.getenv("RAY_WORKER_DATASET_ROOT", "/tmp/ray-dataset"),
    )
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
    overrides_text = str(config_overrides or "").strip()

    if overrides_text:
        def _parse_overrides(text: str) -> Dict[str, Any]:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = yaml.safe_load(text)
                except yaml.YAMLError as exc:
                    raise ValueError(
                        "Failed to parse config_overrides as JSON or YAML"
                    ) from exc
            if parsed is None:
                return {}
            if not isinstance(parsed, dict):
                raise ValueError(
                    "config_overrides must decode to a mapping/dictionary"
                )
            return parsed

        overrides = _parse_overrides(overrides_text)
        if overrides:
            logger.info(
                "Applying config overrides before tuning: %s",
                json.dumps(overrides, indent=2),
            )
            ray_main._deep_update(configs, overrides)
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

    def _propagate_worker_env(ray_cfg: Dict[str, Any]) -> None:
        runtime_env = dict(ray_cfg.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})

        # Propagate common MinIO/MLflow credential keys so Ray workers can
        # materialise datasets without relying on cluster-level secrets.
        for key in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "MINIO_ROOT_USER",
            "MINIO_ROOT_PASSWORD",
            "MLFLOW_S3_ENDPOINT_URL",
            "MLFLOW_TRACKING_URI",
            "MINIO_ENDPOINT",
        ):
            value = os.getenv(key)
            if not value and key == "MLFLOW_TRACKING_URI":
                value = mlflow_cfg.get("tracking_uri")
            if value and key not in env_vars:
                env_vars[key] = value

        endpoint = dataset_info.get("minio_endpoint_used")
        if endpoint and "MLFLOW_S3_ENDPOINT_URL" not in env_vars:
            env_vars["MLFLOW_S3_ENDPOINT_URL"] = str(endpoint)

        if env_vars:
            runtime_env["env_vars"] = env_vars
            ray_cfg["runtime_env"] = runtime_env

    _propagate_worker_env(ray_configs)

    init_kwargs = {
        "address": ray_configs.get("address"),
        "namespace": ray_configs.get("namespace"),
        "runtime_env": ray_configs.get("runtime_env"),
        "ignore_reinit_error": True,
        "local_mode": bool(ray_configs.get("local_mode", False)),
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v}

    def _mask_sensitive(data: Any):
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if any(token in key.lower() for token in ("key", "secret", "password", "token")):
                    masked[key] = "***"
                else:
                    masked[key] = _mask_sensitive(value)
            return masked
        if isinstance(data, list):
            return [_mask_sensitive(item) for item in data]
        return data

    logger.info(
        "Final Ray init kwargs: %s",
        json.dumps(_mask_sensitive(init_kwargs), indent=2),
)
    logger.info("Resolved Ray storage path: %s", storage_path)

    def _serialise_scaling_config(config):
        for attr in ("as_dict", "to_dict", "as_legacy_dict"):
            if hasattr(config, attr):
                method = getattr(config, attr)
                try:
                    return method()
                except TypeError:
                    continue
        if hasattr(config, "__dict__"):
            return config.__dict__
        return str(config)

    scaling_serialisable = _serialise_scaling_config(scaling_config)
    if isinstance(scaling_serialisable, str):
        logger.info("Resolved Ray scaling config: %s", scaling_serialisable)
    else:
        logger.info(
            "Resolved Ray scaling config: %s",
            json.dumps(scaling_serialisable, indent=2),
        )
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


@dsl.component(base_image=BASE_IMAGE)
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
    config_overrides: str = "",

) -> None:
    """Run the final Ray training job using the best configuration."""
    import logging
    import platform
    import sys
    from pathlib import Path as _Path
    import json
    import os
    from typing import Any, Dict
    import yaml

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

    dataset_root = _Path(dataset_artifact.path).resolve()
    local_train_path = dataset_root / "TrainDataset"
    local_test_path = dataset_root / "TestDataset"
    logger.info("Resolved local dataset root: %s", dataset_root)
    logger.info("Resolved training directory: %s", local_train_path)
    logger.info("Resolved test directory: %s", local_test_path)
    discovered_tests = dataset_info.get("test_datasets", [])
    if discovered_tests:
        logger.info("Download step discovered test datasets: %s", discovered_tests)
    try:
        available_test_dirs = [
            p.name for p in local_test_path.iterdir() if p.is_dir()
        ]
    except FileNotFoundError:
        available_test_dirs = []
    logger.info("Test directories available locally: %s", available_test_dirs)
    if not local_train_path.exists():
        logger.warning("Training dataset not found locally at %s", local_train_path)

    remote_train_path = dataset_info.get("train_path") or configs.get("train_path")
    remote_test_path = dataset_info.get("test_path") or configs.get("test_path")
    if not remote_train_path:
        raise ValueError("Remote train_path is required in dataset metadata or config")
    if not remote_test_path:
        logger.warning("Remote test_path missing; falling back to local TestDataset path")
        remote_test_path = str(local_test_path)

    configs["train_path"] = str(remote_train_path)
    configs["test_path"] = str(remote_test_path)
    configs["local_train_dir"] = str(local_train_path)
    configs["local_test_dir"] = str(local_test_path)
    configs["dataset_info"] = dataset_info
    configs.setdefault(
        "worker_dataset_root",
        os.getenv("RAY_WORKER_DATASET_ROOT", "/tmp/ray-dataset"),
    )
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

    overrides_text = str(config_overrides or "").strip()
    overrides: Dict[str, Any] = {}

    if overrides_text:
        def _parse_overrides(text: str) -> Dict[str, Any]:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = yaml.safe_load(text)
                except yaml.YAMLError as exc:
                    raise ValueError(
                        "Failed to parse config_overrides as JSON or YAML"
                    ) from exc
            if parsed is None:
                return {}
            if not isinstance(parsed, dict):
                raise ValueError(
                    "config_overrides must decode to a mapping/dictionary"
                )
            return parsed

        overrides = _parse_overrides(overrides_text)
        if overrides:
            logger.info(
                "Applying config overrides for final training: %s",
                json.dumps(overrides, indent=2),
            )
            ray_main._deep_update(configs, overrides)

    ray_main._deep_update(configs, best_config)

    if overrides:
        logger.info("Reapplying overrides after merging best config")
        ray_main._deep_update(configs, overrides)

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

    def _propagate_worker_env(ray_cfg: Dict[str, Any]) -> None:
        runtime_env = dict(ray_cfg.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})

        for key in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "MINIO_ROOT_USER",
            "MINIO_ROOT_PASSWORD",
            "MLFLOW_S3_ENDPOINT_URL",
            "MLFLOW_TRACKING_URI",
            "MINIO_ENDPOINT",
        ):
            value = os.getenv(key)
            if not value and key == "MLFLOW_TRACKING_URI":
                value = mlflow_cfg.get("tracking_uri")
            if value and key not in env_vars:
                env_vars[key] = value

        endpoint = dataset_info.get("minio_endpoint_used")
        if endpoint and "MLFLOW_S3_ENDPOINT_URL" not in env_vars:
            env_vars["MLFLOW_S3_ENDPOINT_URL"] = str(endpoint)

        if env_vars:
            runtime_env["env_vars"] = env_vars
            ray_cfg["runtime_env"] = runtime_env

    _propagate_worker_env(ray_configs)

    init_kwargs = {
        "address": ray_configs.get("address"),
        "namespace": ray_configs.get("namespace"),
        "runtime_env": ray_configs.get("runtime_env"),
        "ignore_reinit_error": True,
        "local_mode": bool(ray_configs.get("local_mode", False)),
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v}

    def _mask_sensitive(data: Any):
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if any(token in key.lower() for token in ("key", "secret", "password", "token")):
                    masked[key] = "***"
                else:
                    masked[key] = _mask_sensitive(value)
            return masked
        if isinstance(data, list):
            return [_mask_sensitive(item) for item in data]
        return data

    logger.info(
        "Final Ray init kwargs: %s",
        json.dumps(_mask_sensitive(init_kwargs), indent=2),
    )
    logger.info("Resolved Ray storage path: %s", storage_path)

    def _serialise_scaling_config(config):
        for attr in ("as_dict", "to_dict", "as_legacy_dict"):
            if hasattr(config, attr):
                method = getattr(config, attr)
                try:
                    return method()
                except TypeError:
                    continue
        if hasattr(config, "__dict__"):
            return config.__dict__
        return str(config)

    scaling_serialisable = _serialise_scaling_config(scaling_config)
    if isinstance(scaling_serialisable, str):
        logger.info("Resolved Ray scaling config: %s", scaling_serialisable)
    else:
        logger.info(
            "Resolved Ray scaling config: %s",
            json.dumps(scaling_serialisable, indent=2),
        )
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
    import re

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate")

    from pathlib import Path

    summary_path = Path(training_summary_artifact.path)
    logger.info("Reading training summary from %s", summary_path)
    summary_text = summary_path.read_text(encoding="utf-8")
    logger.info("Training summary payload: %s", summary_text)
    summary = json.loads(summary_text)

    raw_dice_score = summary.get("best_dice_score", 0.0)
    try:
        dice_score = float(raw_dice_score)
    except (TypeError, ValueError):
        dice_score = None
        if isinstance(raw_dice_score, str):
            match = re.search(r"([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+)", raw_dice_score)
            if match:
                try:
                    dice_score = float(match.group(0))
                except ValueError:
                    dice_score = None
        if dice_score is None:
            logger.warning(
                "Unable to parse best_dice_score '%s'; defaulting to 0.0", raw_dice_score
            )
            dice_score = 0.0
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
    config_overrides: str = "",

):
    inspect_task = inspect_dataset_component(
        kube_config_path=kube_config_path,
        env_path=env_path,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_secure=minio_secure,
    )

    tune_task = ray_tune_component(
        dataset_artifact=inspect_task.outputs["dataset_artifact"],
        dataset_info_artifact=inspect_task.outputs["dataset_info_artifact"],
        kube_config_path=kube_config_path,
        profile=profile,
        env_path=env_path,
        ray_address=ray_address,
        ray_namespace=ray_namespace,
        ray_storage_path=ray_storage_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint=mlflow_s3_endpoint,
        config_overrides=config_overrides,
    )

    train_task = ray_train_component(
        dataset_artifact=inspect_task.outputs["dataset_artifact"],
        dataset_info_artifact=inspect_task.outputs["dataset_info_artifact"],
        best_config_artifact=tune_task.outputs["best_config_artifact"],
        kube_config_path=kube_config_path,
        profile=profile,
        env_path=env_path,
        ray_address=ray_address,
        ray_namespace=ray_namespace,
        ray_storage_path=ray_storage_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint=mlflow_s3_endpoint,
        config_overrides=config_overrides,
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
        default="ray_segmentation_pipeline_v9.yaml",
        help="Destination path for the compiled pipeline YAML.",
    )
    args = parser.parse_args()
    compile_pipeline(args.output)
    print(f"Pipeline YAML written to {args.output}")