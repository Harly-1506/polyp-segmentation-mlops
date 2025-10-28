import mimetypes
import os
import sys
from pathlib import Path
from typing import Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from tqdm import tqdm


def make_s3_client(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    region: str = "us-east-1",
    verify_ssl: bool = True,
):
    """
    create S3 client
    - endpoint_url: http://127.0.0.1:9000 (port-forward) or http://minio.mlflow.svc.cluster.local:9000 (in-cluster)
    - verify_ssl: false if self-signed cert
    """
    session = boto3.session.Session()
    s3 = session.client(
        service_name="s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        region_name=region,
        verify=verify_ssl,
        config=Config(
            s3={"addressing_style": "path"},
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )
    return s3


def ensure_bucket(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:

        s3.create_bucket(Bucket=bucket)


class TqdmProgress:
    def __init__(self, total_bytes: int, desc: str):
        self.pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=desc)

    def __call__(self, bytes_amount: int):
        self.pbar.update(bytes_amount)

    def close(self):
        self.pbar.close()


def upload_file(
    s3,
    file_path: Path,
    bucket: str,
    key: str,
    content_type: Optional[str] = None,
    multipart_threshold_mb: int = 32,
    max_concurrency: int = 8,
):
    # guess content type if not provided
    if not content_type:
        content_type, _ = mimetypes.guess_type(str(file_path))
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    file_size = file_path.stat().st_size
    progress = TqdmProgress(file_size, desc=f"Upload {file_path.name}")

    transfer_cfg = TransferConfig(
        multipart_threshold=multipart_threshold_mb * 1024 * 1024,
        max_concurrency=max_concurrency,
        multipart_chunksize=8 * 1024 * 1024,
        use_threads=True,
    )

    try:
        s3.upload_file(
            Filename=str(file_path),
            Bucket=bucket,
            Key=key,
            ExtraArgs=extra_args,
            Callback=progress,
            Config=transfer_cfg,
        )
    finally:
        progress.close()


def upload_directory(
    s3,
    local_dir: Path,
    bucket: str,
    prefix: str = "",
    ignore_hidden: bool = True,
):
    local_dir = local_dir.resolve()
    ensure_bucket(s3, bucket)

    files = [p for p in local_dir.rglob("*") if p.is_file()]
    for fp in files:
        if ignore_hidden and any(
            part.startswith(".") for part in fp.relative_to(local_dir).parts
        ):
            continue
        # compute key
        rel = fp.relative_to(local_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
        upload_file(s3, fp, bucket, key)


def main(dataset_dir: Optional[str] = None) -> None:
    """
    Config MinIO/MLflow S3 backend and upload dataset

    """
    endpoint = "http://127.0.0.1:9000"
    access_key = (
        os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER") or "admin"
    )
    secret_key = (
        os.getenv("AWS_SECRET_ACCESS_KEY")
        or os.getenv("MINIO_ROOT_PASSWORD")
        or "admin1234"
    )
    region = "us-east-1"
    verify_ssl = "false"

    bucket = "dataset"
    prefix = "polyp_dataset"

    if not dataset_dir.exists():
        print(f"[!] Did not see dataset dir: {dataset_dir}")
        sys.exit(1)

    print(f"Endpoint: {endpoint}")
    print(f"Bucket  : {bucket}")
    print(f"Prefix  : {prefix}")
    print(f"Upload  : {dataset_dir}")

    s3 = make_s3_client(
        endpoint, access_key, secret_key, region=region, verify_ssl=verify_ssl
    )
    upload_directory(s3, dataset_dir, bucket=bucket, prefix=prefix)
    print("✅ Upload completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload dataset to MinIO/MLflow S3 backend"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the local dataset directory",
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    main(dataset_dir=dataset_dir)
