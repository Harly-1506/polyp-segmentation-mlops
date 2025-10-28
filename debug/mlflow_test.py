import os

import mlflow

# Thiết lập tracking URI (nếu chưa đặt bằng biến môi trường)
mlflow.set_tracking_uri("http://localhost:5001")

# Tạo file mẫu để lưu artifact
artifact_file = "hello_minio.txt"
with open(artifact_file, "w") as f:
    f.write("✅ Hello from MLflow to MinIO!")

# Tạo experiment nếu chưa có
experiment_name = "minio_test_experiment"
mlflow.set_experiment(experiment_name)

# Start 1 run và log artifact
with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")
    mlflow.log_param("example_param", 42)
    mlflow.log_artifact(artifact_file)
    print("✅ Artifact logged successfully.")
