from kfp import compiler, dsl


@dsl.component(
    base_image="rayproject/ray:2.48.0-py312-cpu",
    packages_to_install=["mlflow==3.1.0", "pyarrow>=16.1,<21", "torch==2.7.1"],
)
def ray_torch_mlflow_smoke_test(
    # Bạn vẫn có thể override bằng tham số pipeline nếu muốn
    ray_address: str = "ray://kuberay-raycluster-head-svc.development.svc.cluster.local:10001",
    mlflow_tracking_uri: str = "http://mlflow-tracking-service.mlflow.svc.cluster.local:5000",
    experiment_name: str = "kubeflow-ray-smoke",
    ray_namespace: str = "default",
) -> str:
    import json
    import os
    import platform
    import socket
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    from urllib.parse import urlparse

    # ======== HARD-CODE ENV FOR SMOKE TEST ========
    # Nếu muốn đổi, sửa 4 dòng dưới:
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-tracking-service.mlflow.svc.cluster.local:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.mlflow.svc.cluster.local:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "admin1234"
    # Khuyến nghị thêm:
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    os.environ["MLFLOW_S3_IGNORE_TLS"] = os.environ.get("MLFLOW_S3_IGNORE_TLS", "true")

    def p(msg: str) -> None:
        print(msg, flush=True)

    def _json_roundtrip(obj):
        return json.loads(json.dumps(obj))

    # ======== DRIVER ENV INFO ========
    p("=== DRIVER ENV INFO (before Ray connect) ===")
    p(f"python: {sys.executable}")
    p(f"python_version: {sys.version}")
    p(f"platform: {platform.platform()}")
    p(f"ray_address(effect): {ray_address}")
    p(f"ray_namespace(effect): {ray_namespace}")
    p(f"MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    p(f"MLFLOW_S3_ENDPOINT_URL: {os.environ.get('MLFLOW_S3_ENDPOINT_URL')}")
    try:
        pf = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        pf = f"<pip freeze failed: {e}>"
    for line in pf.splitlines()[:50]:
        p("  " + line)

    try:
        import torch as _t
        drv_has_torch, drv_torch_detail = True, _t.__version__
    except Exception as e:
        drv_has_torch, drv_torch_detail = False, f"import torch failed on driver: {e}"
    p(f"driver_has_torch={drv_has_torch}; detail={drv_torch_detail}")

    import ray

    # ======== Preflight TCP đến Ray client ========
    u = urlparse(ray_address)
    host = u.hostname or ray_address
    port = u.port or 10001
    p(f"=== Preflight: TCP connect to {host}:{port} ===")
    try:
        s = socket.create_connection((host, port), timeout=5)
        s.close()
        p("Preflight OK: TCP reachable.")
    except Exception as e:
        p(f"[FATAL] Cannot reach Ray client at {host}:{port}\nError: {e}")
        raise RuntimeError("Ray TCP preflight failed") from e

    # ======== Connect Ray & propagate ENV sang workers ========
    env_to_propagate = {
        k: os.environ[k]
        for k in [
            "MLFLOW_TRACKING_URI",
            "MLFLOW_S3_ENDPOINT_URL",
            "MLFLOW_S3_IGNORE_TLS",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
            "AWS_EC2_METADATA_DISABLED",
        ]
        if k in os.environ
    }

    p("=== Connecting to Ray Cluster ===")
    ray.init(
        address=ray_address,
        namespace=ray_namespace,
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env={"env_vars": env_to_propagate},
    )
    p("Connected to Ray.")
    p(f"Ray version: {ray.__version__}")
    p(f"Ray cluster resources: {ray.cluster_resources()}")

    # ========== PROBES ==========
    @ray.remote(max_retries=0)
    def ping():
        return "pong"

    p("=== Ray ping ===")
    p(f"ping: {ray.get(ping.remote())}")

    @ray.remote(max_retries=0)
    def py_probe():
        import platform as _pl
        import sys as _s
        return _json_roundtrip({"python": _s.version, "executable": _s.executable, "platform": _pl.platform()})

    p("=== Python probe on worker ===")
    py_env = ray.get(py_probe.remote())
    p(json.dumps(py_env, indent=2))

    @ray.remote(max_retries=0)
    def numpy_probe():
        import numpy as np
        import numpy.core.multiarray as _ma
        return _json_roundtrip({"numpy_version": str(np.__version__), "numpy_file": str(getattr(np, "__file__", ""))})

    p("=== NumPy probe on worker ===")
    np_env = ray.get(numpy_probe.remote())
    p(json.dumps(np_env, indent=2))

    @ray.remote(max_retries=0)
    def torch_probe():
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
            return _json_roundtrip({
                "torch_ok": True,
                "torch_version": str(torch.__version__),
                "cuda_available": cuda,
                "cuda_device_count": int(torch.cuda.device_count() if cuda else 0),
                "mean": float((torch.randn(8, 4) @ torch.randn(4, 3)).mean().item()),
            })
        except Exception as e:
            return _json_roundtrip({"torch_ok": False, "error": f"{type(e).__name__}: {e}"})

    p("=== Torch probe on worker ===")
    t_probe = ray.get(torch_probe.remote())
    p(json.dumps(t_probe, indent=2))
    if not t_probe.get("torch_ok", False):
        raise RuntimeError(f"Torch probe failed: {t_probe.get('error')}")

    # ========== Tiny trainer (best effort) ==========
    trainer_result, trainer_error = None, None
    try:
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer

        def train_loop_per_worker():
            import torch
            m = torch.nn.Linear(10, 2)
            opt = torch.optim.SGD(m.parameters(), lr=0.01)
            loss_fn = torch.nn.CrossEntropyLoss()
            x = torch.randn(16, 10)
            y = torch.randint(0, 2, (16,))
            for _ in range(2):
                opt.zero_grad()
                out = m(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
            return {"final_loss": float(loss.detach().cpu().item())}

        use_gpu = bool(t_probe.get("cuda_available"))
        trainer = TorchTrainer(
            train_loop_per_worker,
            scaling_config=ScalingConfig(num_workers=1, use_gpu=use_gpu, trainer_resources={"CPU": 0}),
        )
        trainer_result = trainer.fit()
        p(f"TorchTrainer result: {trainer_result}")
    except Exception as e:
        trainer_error = str(e)
        p(f"TorchTrainer failed: {trainer_error}")

    # ========== MLflow logging ==========
    import mlflow
    p("=== Logging to MLflow ===")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="ray-torch-mlflow-smoke") as run:
        mlflow.log_param("ray_address", ray_address)
        mlflow.log_param("ray_namespace", ray_namespace)
        mlflow.log_param("driver_has_torch", drv_has_torch)
        mlflow.log_param("driver_torch_detail", drv_torch_detail)

        for k in ["python", "executable", "platform"]:
            mlflow.log_param("worker_" + k, str(py_env.get(k)))
        mlflow.log_param("worker_numpy_version", np_env.get("numpy_version"))
        mlflow.log_param("worker_numpy_file", np_env.get("numpy_file"))

        for k in ["torch_ok", "torch_version", "cuda_available", "cuda_device_count", "mean"]:
            if k in t_probe:
                mlflow.log_param("worker_" + k, str(t_probe[k]))

        if trainer_result and getattr(trainer_result, "metrics", None):
            try:
                mlflow.log_metric("trainer_final_loss", float(trainer_result.metrics.get("final_loss", -1)))
            except Exception:
                pass

        # Ghi debug payload làm artifact (client-side → dùng AWS_* ở trên)
        payload = {
            "driver": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "pip_freeze": pf},
            "ray": {"version": ray.__version__, "cluster_resources": ray.cluster_resources()},
            "py_env": py_env,
            "numpy_env": np_env,
            "torch_probe": t_probe,
            "trainer_error": trainer_error,
        }
        import tempfile
        fd, tmp = tempfile.mkstemp(prefix="ray_torch_mlflow_debug_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        mlflow.log_artifact(tmp, artifact_path="debug")
        p(f"MLflow run id: {run.info.run_id}")

    return json.dumps({"worker_torch_ok": bool(t_probe.get("torch_ok", False))})


@dsl.pipeline(name="ray-torch-mlflow-smoke-pipeline")
def pipeline(
    ray_address: str = "ray://kuberay-raycluster-head-svc.development.svc.cluster.local:10001",
    mlflow_tracking_uri: str = "http://mlflow-tracking-service.mlflow.svc.cluster.local:5000",
    experiment_name: str = "kubeflow-ray-smoke",
    ray_namespace: str = "default",
):
    # Không cần map ENV từ ConfigMap/Secret ở smoke — đã hard-code trong component
    ray_torch_mlflow_smoke_test(
        ray_address=ray_address,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        ray_namespace=ray_namespace,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="ray_torch_mlflow_smoke_pipeline_hardenv.yaml")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    if args.compile:
        compiler.Compiler().compile(pipeline_func=pipeline, package_path=args.output)
        print("Wrote:", args.output)
