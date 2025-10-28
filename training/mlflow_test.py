import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.create_experiment(
    "/Users/harlystudy1506@gmail.com/check-databricks-connection",
    artifact_location="dbfs:/Volumes/test/mlflow/check-databricks-connection",
)
mlflow.set_experiment("/Users/harlystudy1506@gmail.com/check-databricks-connection")

with mlflow.start_run():
    mlflow.log_metric("foo", 1)
    mlflow.log_metric("bar", 2)
