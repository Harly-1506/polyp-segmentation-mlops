import os

from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

import mlflow
import ray
from ray import train


# Viết file giả để test artifact
def create_test_file():
    output_dir = "output_logs"
    os.makedirs(output_dir, exist_ok=True)  # tạo thư mục nếu chưa có

    with open(os.path.join(output_dir, "test_output.txt"), "w") as f:
        f.write("Log from Ray worker")

# Hàm train giả định cho Ray
def train_fn(config):
    # Lấy context để kiểm tra rank
    rank = train.get_context().get_world_rank()
    
    # Chỉ log từ worker rank 0
    if rank == 0:
        print(f"Worker {rank}: Logging to MLflow...")
        mlflow.start_run()
        mlflow.log_param("dummy_param", 123)
        mlflow.log_metric("accuracy", 0.95)
        print("Current working dir:", os.getcwd())
        print("File exists?", os.path.exists("output_logs/test_output.txt"))    
        mlflow.log_artifact("output_logs/test_output.txt")
        mlflow.end_run()
    else:
        print(f"Worker {rank}: Skip MLflow logging")

# === Khởi chạy chính ===
if __name__ == "__main__":
    # Cấu hình biến môi trường MLflow + MinIO
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "mlflow123"

    # Tạo file test để log artifact
    create_test_file()

    # Khởi tạo Ray runtime (nếu cần)
    ray.init()

    # Cấu hình training (số worker: 2 để thấy phân tán)
    scaling_config = ScalingConfig(num_workers=2, use_gpu=False)

    # Dùng Ray Train để chạy train_fn
    trainer = TorchTrainer(train_loop_per_worker=train_fn, scaling_config=scaling_config)
    result = trainer.fit()

    ray.shutdown()
