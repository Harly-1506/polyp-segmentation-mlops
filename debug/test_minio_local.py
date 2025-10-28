#!/usr/bin/env python3
# scripts/test_minio_dataset.py
import os
import sys
import json
import yaml
import argparse
import logging
import platform
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import boto3
from botocore.client import Config as BotoConfig
from dotenv import dotenv_values

# Use non-interactive backend for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_endpoint(cli_endpoint: str, env_values: Dict[str, str], mlflow_cfg: Dict[str, Any], secure: bool) -> str:
    endpoint = (
        cli_endpoint
        or env_values.get("MINIO_ENDPOINT")
        or env_values.get("AWS_ENDPOINT_URL")  # allow direct AWS_ENDPOINT_URL
        or (mlflow_cfg.get("s3_endpoint_url") if isinstance(mlflow_cfg, dict) else None)
        or (mlflow_cfg.get("s3_endpoint") if isinstance(mlflow_cfg, dict) else None)
    )
    if endpoint and not str(endpoint).startswith("http"):
        scheme = "https" if secure else "http"
        endpoint = f"{scheme}://{str(endpoint).lstrip('/')}"
    return endpoint or ""


def first_existing_prefix(s3_client, bucket: str, candidates: List[str]) -> str:
    for cand in candidates:
        cand = cand.rstrip("/")
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=cand + "/", MaxKeys=1)
        if int(resp.get("KeyCount", 0)) > 0:
            return cand
    raise FileNotFoundError(f"No training dataset prefix found; checked: {candidates}")


def has_nonempty_dir(s3_client, bucket: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/") + "/"
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return int(resp.get("KeyCount", 0)) > 0


def collect_samples(s3_client, bucket: str, image_prefix: str, limit: int = 3) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    image_prefix = image_prefix.rstrip("/") + "/"
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=image_prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            mask_key = key.replace("/images/", "/masks/", 1)
            try:
                s3_client.head_object(Bucket=bucket, Key=mask_key)
            except Exception:
                # mask missing, skip
                continue
            samples.append(
                {
                    "image_key": key,
                    "mask_key": mask_key,
                    "image_uri": f"s3://{bucket}/{key}",
                    "mask_uri": f"s3://{bucket}/{mask_key}",
                }
            )
            if len(samples) >= limit:
                return samples
    return samples


def save_preview_png(s3_client, bucket: str, samples: List[Dict[str, str]], out_path: Path) -> None:
    if not samples:
        logging.warning("No (image, mask) pairs found to preview; skipping preview.png")
        return
    rows = len(samples)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows))
    if rows == 1:
        axes = [axes]
    for i, s in enumerate(samples):
        img_obj = s3_client.get_object(Bucket=bucket, Key=s["image_key"])
        msk_obj = s3_client.get_object(Bucket=bucket, Key=s["mask_key"])
        img = Image.open(BytesIO(img_obj["Body"].read())).convert("RGB")
        msk = Image.open(BytesIO(msk_obj["Body"].read()))

        axes[i][0].imshow(img)
        axes[i][0].set_title(f"Image #{i+1}")
        axes[i][0].axis("off")

        axes[i][1].imshow(msk, cmap="gray")
        axes[i][1].set_title(f"Mask #{i+1}")
        axes[i][1].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logging.info("Saved preview image to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Local smoke test for MinIO dataset layout (matches download_dataset logic).")
    parser.add_argument("--kube-config", default="training/configs/kube_configs.yaml", help="Path to kube_configs.yaml")
    parser.add_argument("--env", dest="env_path", default=".env", help="Path to .env to load AWS/MinIO creds")
    parser.add_argument("--output-dir", default="out_minio_test", help="Where to write dataset_info.json and preview.png")
    parser.add_argument("--minio-endpoint", default="", help="Override MinIO endpoint (e.g. http://127.0.0.1:9000)")
    parser.add_argument("--access-key", default="", help="Override AWS_ACCESS_KEY_ID")
    parser.add_argument("--secret-key", default="", help="Override AWS_SECRET_ACCESS_KEY")
    parser.add_argument("--secure", action="store_true", help="Use https if endpoint is host:port")
    parser.add_argument("--samples", type=int, default=3, help="How many (image,mask) pairs to preview")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Environment: python=%s platform=%s", sys.version.replace("\n", " "), platform.platform())

    cfg = load_yaml(args.kube_config)
    bucket = cfg.get("bucket")
    prefix = (cfg.get("dataset_prefix") or "").rstrip("/")
    mlflow_cfg = cfg.get("mlflow", {})
    if not bucket or not prefix:
        raise SystemExit("ERROR: 'bucket' and 'dataset_prefix' must exist in kube_configs.yaml")

    # Load .env then overlay CLI
    env_values = {}
    env_file = Path(args.env_path)
    if env_file.exists():
        env_values = {k: str(v) for k, v in dotenv_values(env_file).items() if v is not None}
        for k, v in env_values.items():
            os.environ.setdefault(k, v)
        logging.info("Loaded %d env vars from %s", len(env_values), env_file)

    access_key = args.access_key or os.environ.get("AWS_ACCESS_KEY_ID") or env_values.get("AWS_ACCESS_KEY_ID") or env_values.get("MINIO_ROOT_USER")
    secret_key = args.secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY") or env_values.get("AWS_SECRET_ACCESS_KEY") or env_values.get("MINIO_ROOT_PASSWORD")
    endpoint = resolve_endpoint(args.minio_endpoint, env_values, mlflow_cfg, args.secure)

    if not endpoint:
        raise SystemExit("ERROR: Could not resolve MinIO endpoint. Pass --minio-endpoint or set MINIO_ENDPOINT / AWS_ENDPOINT_URL or configure mlflow.s3_endpoint_url in kube_configs.yaml")
    if not access_key or not secret_key:
        raise SystemExit("ERROR: Missing credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY). Provide via CLI or .env")

    logging.info("Using endpoint: %s", endpoint)
    logging.info("Using bucket: %s, dataset_prefix: %s", bucket, prefix)

    # boto3 clients
    session = boto3.session.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    s3_client = session.client("s3", endpoint_url=endpoint, config=BotoConfig(signature_version="s3v4"), region_name=session.region_name or "us-east-1")

    # Discover train prefix
    try:
        train_prefix = first_existing_prefix(s3_client, bucket, [f"{prefix}/TrainDataset", f"{prefix}/Traindataset"])
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(2)

    images_prefix = f"{train_prefix}/images"
    masks_prefix  = f"{train_prefix}/masks"

    has_images = has_nonempty_dir(s3_client, bucket, images_prefix)
    has_masks  = has_nonempty_dir(s3_client, bucket, masks_prefix)

    # This check mirrors your AssertionError location
    if not has_images or not has_masks:
        logging.error("Missing train dirs (or empty): %s exists=%s | %s exists=%s",
                      images_prefix, has_images, masks_prefix, has_masks)
        logging.error("This would trigger: AssertionError: Missing train dirs: s3://%s/%s / s3://%s/%s",
                      bucket, images_prefix, bucket, masks_prefix)
        sys.exit(3)

    # Discover if TestDataset exists (optional)
    test_prefix = f"{prefix}/TestDataset"
    has_test = has_nonempty_dir(s3_client, bucket, test_prefix)
    if not has_test:
        logging.warning("No testing files found under prefix %s", test_prefix)

    # Collect a few sample pairs for preview
    samples = collect_samples(s3_client, bucket, images_prefix, limit=max(0, int(args.samples)))
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if samples:
        save_preview_png(s3_client, bucket, samples, out_dir / "preview.png")
    else:
        logging.warning("Could not find any (image,mask) pairs to preview from %s", images_prefix)

    # Write dataset_info.json like your component
    info = {
        "bucket": bucket,
        "prefix": prefix,
        "train_path": f"s3://{bucket}/{train_prefix.strip('/')}",
        "test_path": f"s3://{bucket}/{test_prefix.strip('/')}",
        "s3_endpoint": endpoint,
        "sample_uris": samples,
    }
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    logging.info("Wrote dataset_info.json to %s", out_dir / "dataset_info.json")

    # Extra: show env that Ray workers will need
    logging.info("Ray workers should have env: AWS_ACCESS_KEY_ID (set=%s), AWS_SECRET_ACCESS_KEY (set=%s), AWS_ENDPOINT_URL=%s",
                 bool(access_key), bool(secret_key), endpoint)

    logging.info("SUCCESS: Dataset layout looks good. Use train_path=%s in Ray configs.",
                 info["train_path"])


if __name__ == "__main__":
    main()
