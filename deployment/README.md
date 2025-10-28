# Development & Deployment Playbook

This folder provides Kubernetes assets and guidance for deploying the Triton inference stack to GKE with KServe, instrumented with Prometheus and Grafana.

## Quickstart: Deploying the Triton stack to GKE

Follow the shared preparation steps below and then choose the CPU-only or GPU-enabled rollout depending on the node pools available in your cluster.

### Shared preparation

1. **Authenticate and configure gcloud**
   ```bash
   export PROJECT_ID="polyp-mlops-1506"
   export REGION="asia-southeast1"
   export ARTIFACT_REPO="polyp-inference"

   gcloud auth login
   gcloud config set project "${PROJECT_ID}"
   gcloud config set compute/region "${REGION}"
   # Enable core GKE/Artifact Registry services (requires Service Usage Admin or Project Owner)
   gcloud services enable container.googleapis.com artifactregistry.googleapis.com compute.googleapis.com
   ```

   > **Permission denied?** The caller must hold
   > [`roles/serviceusage.serviceUsageAdmin`](https://cloud.google.com/service-usage/docs/reference/rest/v1/operations)
   > (or Project Owner/Editor). Grant the role before rerunning the enablement
   > command when you see `AUTH_PERMISSION_DENIED`. Service accounts used by
   > Terraform or Jenkins require the same role if they bootstrap new projects.

2. **Create an Artifact Registry to host the FastAPI gateway image**
   ```bash
   gcloud artifacts repositories create "${ARTIFACT_REPO}" \
     --repository-format=docker \
     --location="${REGION}" \
     --description="Containers for Triton gateway"
   ```

3. **Build and push the Triton gateway** using the Dockerfile under `app/`:
   ```bash
   GATEWAY_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/polyp-gateway:latest"

   docker build -t "${GATEWAY_IMAGE}" -f app/Dockerfile .
   gcloud auth configure-docker "${REGION}-docker.pkg.dev"
   docker push "${GATEWAY_IMAGE}"
   ```
   The container reads its configuration from environment variables defined in
   `app/backend/config.py`, including `APP_TRITON_URL`, `APP_TRITON_MODEL_NAME`,
   and `APP_REQUEST_TIMEOUT_SECONDS`. When running in Kubernetes, point
   `APP_TRITON_URL` at the predictor service DNS name
   (for example `http://polyp-segmentation-predictor.kserve-inference.svc.cluster.local:80`).

4. **Export a model artifact for Triton** using `training/scripts/export_to_onnx.py`:
   ```bash
   CHECKPOINT="training/checkpoints/UNet/Unet81PolypPVT-best.pth"
   ONNX_EXPORT="artifacts/polyp-segmentation/model.onnx"

   uv run python -m training.scripts.export_to_onnx "${CHECKPOINT}" "${ONNX_EXPORT}" --image-size 256 --dynamic
   ```
   Upload the exported ONNX file to a bucket accessible from Triton:
   ```bash
   MODEL_BUCKET="gs://my-polyp-models"
   gsutil mb -l "${REGION}" "${MODEL_BUCKET}"
   gsutil cp "${ONNX_EXPORT}" "${MODEL_BUCKET}/polyp-segmentation/onnx/1/model.onnx"
   ```

5. **Update the manifests** under `deployment/kserve/` with your resources:
   - Set `storageUri` to the root of the Triton model repository (for example `gs://.../polyp-segmentation/onnx`). Triton automatically serves the highest numeric version folder.
   - Point each transformer container `image` to `"${GATEWAY_IMAGE}"`.
   - Configure `APP_TRITON_URL` to reach the predictor service on port 80.
   - If the model bucket is private, configure Workload Identity and set `serviceAccountName` appropriately.

### CPU-only rollout

1. **Provision a CPU-focused cluster**

   - _Terraform_: customise `terraform/terraform.tfvars` (project, region, node pool sizing, image, etc.), then run `terraform init && terraform apply`.
   - _gcloud_:
     ```bash
     gcloud container clusters create polyp-triton \
       --zone "${REGION}-a" \
       --machine-type n1-standard-4 \
       --num-nodes 3 \
       --release-channel regular \
       --enable-ip-alias
     gcloud container clusters get-credentials polyp-triton --zone "${REGION}-a" --project "${PROJECT_ID}"
     ```

     ```
     Ngăn lỗi token hết hạn / endpoint thay đổ
     gcloud container clusters get-credentials ml-inference-cluster \
  --region asia-southeast1 \
  --project polyp-mlops-1506


     ```

2. **Install cert-manager and KServe** once the cluster is ready:
   ```bash
   CERT_MANAGER_VERSION="v1.14.4"
   kubectl apply -f "https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml"
   kubectl wait --for=condition=Available deployment/cert-manager -n cert-manager --timeout=180s
   kubectl wait --for=condition=Available deployment/cert-manager-cainjector -n cert-manager --timeout=180s
   kubectl wait --for=condition=Available deployment/cert-manager-webhook -n cert-manager --timeout=180s

   KSERVE_VERSION="v0.11.2"
   kubectl apply -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"
   kubectl wait --for=condition=Available deployment -l control-plane=kserve-controller-manager -n kserve --timeout=180s
   ```
   If any wait command times out, ensure the pods are scheduled before retrying.


3. **Prepare and deploy the manifests** built from the shared steps above:
   - Update any remaining placeholders before applying:
     - `storageUri`: point to the exported ONNX model in GCS / S3 (reference the model repository root, not a specific version).
     - `image`: point to the Triton gateway image you pushed to Artifact Registry.
     - `APP_TRITON_URL`: target the Triton predictor service inside the cluster, e.g. `http://polyp-segmentation-predictor.kserve-inference.svc.cluster.local:80`.
     - For the Gradio UI kustomization under `deployment/ui` set `image` to the published UI container and `BACKEND_URL` to the FastAPI transformer service (for example `http://polyp-segmentation-transformer.kserve-inference.svc.cluster.local:8081`).
   - Apply the CPU baseline:
     ```bash
     kubectl create namespace kserve-inference
     kubectl apply -k deployment/kserve
     kubectl apply -f deployment/kserve/inferenceservice-basic-cpu.yaml -n kserve-inference
     kubectl apply -k deployment/ui
     kubectl get inferenceservices -n kserve-inference
     ```
4. **Validate the deployment** by port-forwarding the transformer service:
   ```bash
   kubectl port-forward svc/polyp-segmentation-transformer 9000:80 -n kserve-inference
   BACKEND_URL="http://localhost:9000" python app/backend/tests/local_smoke_test.py
   ```
   Inspect the transformer logs (`kubectl logs deploy/polyp-segmentation-transformer-default -n kserve-inference -c gateway`) for request traces and confirm `/metrics` reports HTTP 200.

### GPU-enabled rollout

1. **Provision a GPU-capable cluster**

   - _Terraform_: enable the GPU node pool variables (machine type, accelerator type/count) in `terraform/terraform.tfvars` before applying.
   - _gcloud_:
     ```bash
     gcloud container clusters create polyp-triton-gpu \
       --zone "${REGION}-a" \
       --machine-type n1-standard-8 \
       --accelerator type=nvidia-tesla-t4,count=1 \
       --num-nodes 3 \
       --release-channel regular \
       --enable-ip-alias
     gcloud container clusters get-credentials polyp-triton-gpu --zone "${REGION}-a" --project "${PROJECT_ID}"
     kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
     ```
     Wait for the `nvidia-driver-installer` DaemonSet pods to report `Ready` before moving on.

2. **Install cert-manager and KServe** (same commands as the CPU path).

3. **Deploy the base kustomization and GPU InferenceService**:
   ```bash
   kubectl create namespace kserve-inference --dry-run=client -o yaml | kubectl apply -f -
   kubectl apply -k deployment/kserve
   kubectl apply -f deployment/kserve/inferenceservice-basic-gpu.yaml -n kserve-inference
   kubectl get inferenceservices -n kserve-inference
   ```

4. **Validate GPU access**:
   ```bash
   kubectl exec -it deploy/polyp-segmentation-predictor-default -n kserve-inference -- nvidia-smi
   kubectl port-forward svc/polyp-segmentation-gpu-transformer 9000:80 -n kserve-inference
   BACKEND_URL="http://localhost:9000" python app/backend/tests/local_smoke_test.py
   ```
   Review the predictor logs to ensure Triton binds the GPU and the `nvidia.com/gpu` resource is consumed.

After either rollout succeeds you can proceed with canary updates, monitoring, and automation as described later in this guide.

## 1. Local validation

The steps below walk through exporting a checkpoint, standing up a Triton
server, and wiring the FastAPI gateway to it for end-to-end smoke tests. The
same process works on Linux and macOS (Apple Silicon requires the
`arm64v8/tritonserver` image).

### 1.1 Prerequisites

- Docker Desktop or the Docker Engine CLI.
- (Optional) NVIDIA Container Toolkit when you want GPU acceleration.
- `uv` for dependency management (install from https://docs.astral.sh/uv/).

### 1.2 Export the model to ONNX

```bash
uv run python -m training.scripts.export_to_onnx \
  training/checkpoints/UNet/Unet81PolypPVT-best.pth \
  artifacts/polyp-segmentation/model.onnx \
  --image-size 256 --dynamic
```

### 1.3 Prepare a Triton model repository

```bash
export MODEL_REPO=$(pwd)/deployment/local-triton/models
mkdir -p "${MODEL_REPO}/polyp-segmentation/1"
cp artifacts/polyp-segmentation/model.onnx "${MODEL_REPO}/polyp-segmentation/1/model.onnx"
cat <<'EOF' > "${MODEL_REPO}/polyp-segmentation/config.pbtxt"
name: "polyp-segmentation"
platform: "onnxruntime_onnx"
max_batch_size: 1
input {
  name: "input"
  data_type: TYPE_FP32
  format: FORMAT_NCHW
  dims: [3, 256, 256]
}
output {
  name: "output"
  data_type: TYPE_FP32
  dims: [1, 256, 256]
}
instance_group [{ kind: KIND_CPU }]
EOF
```

Adjust `instance_group` to `KIND_GPU` and add `gpus: [0]` when running on a
machine with NVIDIA GPUs. Update the tensor shapes if your export uses dynamic
dimensions (for example `[3, -1, -1]` and `[1, -1, -1]`). Triton will reject
the model if the configuration mismatches the ONNX metadata.

### 1.4 Run Triton locally

```bash
export APP_TRITON_MODEL_VERSION="1"  
docker run --rm --name triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${MODEL_REPO}:/models" nvcr.io/nvidia/tritonserver:24.10-py3 \
  tritonserver --model-repository=/models

```

Triton will load the ONNX model and expose HTTP/gRPC/metrics endpoints on
ports 8000, 8001, and 8002 respectively. Check the container logs for `READY`
messages before continuing and confirm readiness via:

```bash
curl http://localhost:8000/v2/health/ready
```

### 1.5 Build and run the FastAPI gateway

```bash
docker build -t polyp-gateway:local -f app/Dockerfile .
uv sync --group inference
export APP_TRITON_URL="http://localhost:8000"
uv run --group inference -- uvicorn app.backend.main:app --reload --port 8081
```

If you prefer Docker Compose, update `.env` with `APP_TRITON_URL=http://triton:8000` and run
`docker compose -f docker-compose-app.yaml up --build`.

### 1.6 Smoke tests and UI
```bash
export APP_TRITON_URL="http://localhost:8000"
export APP_TRITON_MODEL_NAME="polyp-segmentation"
uv run --group inference -- uvicorn app.backend.main:app --host 0.0.0.0 --port 8081
export BACKEND_URL="http://localhost:8081"
uv run --group inference -- python app/frontend/gradio_ui.py
```

Both commands should succeed with the mock client disabled, confirming that the
gateway can reach the locally running Triton server.

Before building a release image, run a quick HTTP check against the FastAPI
gateway to verify that Triton responds end-to-end:

```bash
curl -X POST -F "file=@app/backend/8.png" http://localhost:8081/predict
```

The response should include the encoded mask payload, mirroring the output of
`app/backend/tests/local_smoke_test.py`.

## 2. Build once, deploy anywhere

Push the Docker image produced above to your registry and set the `image` fields inside the manifests under `deployment/kserve`.

```bash
export PROJECT_ID=polyp-mlops-1506
export REGION=asia-southeast1
export CLUSTER=ml-inference-cluster
export NAMESPACE=kserve-inference
export BUCKET=my-polyp-models
export GSA=kserve-infer-sa
export KSA=kserve-model-sa
export MODEL_DIR=polyp-segmentation

gcloud container clusters update $CLUSTER --region $REGION \
  --workload-pool=${PROJECT_ID}.svc.id.goog

gcloud iam service-accounts create $GSA --project $PROJECT_ID
gsutil iam ch serviceAccount:${GSA}@${PROJECT_ID}.iam.gserviceaccount.com:roles/storage.objectViewer gs://${BUCKET}
kubectl -n $NAMESPACE create serviceaccount $KSA
gcloud iam service-accounts add-iam-policy-binding \
  "${GSA}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/iam.workloadIdentityUser" \
  --member "serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/${KSA}]"
kubectl -n $NAMESPACE annotate serviceaccount $KSA \
  iam.gke.io/gcp-service-account=${GSA}@${PROJECT_ID}.iam.gserviceaccount.com --overwrite


```

```bash
kubectl create namespace kserve-inference
kubectl apply -k deployment/kserve
kubectl apply -k deployment/ui
```

Update placeholders:
- `storageUri`: point to the exported ONNX model in GCS / S3 (reference the model repository root, not a specific version).
- `image`: point to the gateway image built by CI.
- `APP_TRITON_URL`: target the Triton predictor service inside the cluster, e.g. `http://polyp-segmentation-predictor.kserve-inference.svc.cluster.local:80`.

For the Gradio UI deployment under `deployment/ui` update:
- `image`: to the published UI container (can reuse the gateway image when started with `RUN_UI=true` and `RUN_BACKEND=false`).
- `BACKEND_URL`: to the Kubernetes service that fronts the FastAPI transformer (for example `http://polyp-segmentation-gpu-transformer.kserve-inference.svc.cluster.local:8081`).

## 4. Monitoring Stack

Install kube-prometheus-stack and Grafana in the `observability` namespace. The provided Helm values assume a GKE cluster that exposes the default `standard-rwo` storage class and that KServe is installed in `kserve-inference`:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts

helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace observability --create-namespace \
  -f deployment/observability/prometheus/values.yaml

helm upgrade --install grafana grafana/grafana \
  --namespace observability \
  -f deployment/observability/grafana/values.yaml

kubectl apply -k deployment/observability
```

If your GKE cluster uses a different storage class, override `prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName` and `persistence.storageClassName` for Grafana during the Helm upgrades. The Prometheus values include an additional scrape job that targets the FastAPI gateway Service; update the hostname if your Service name or namespace differs. The Kustomize overlay creates the namespace, registers the `ServiceMonitor`, loads the polyp dashboard into Grafana via ConfigMap, and applies alerting rules for latency and health probes. After installation you can reach Grafana either by enabling an ingress in the Helm values or by port-forwarding:

```bash
kubectl -n observability port-forward svc/grafana 3000:80
```

Log in with the admin credentials defined in `values.yaml` (or retrieve the autogenerated password with `kubectl get secret grafana -n observability -o jsonpath='{.data.admin-password}' | base64 -d`) and select the “Polyp Segmentation” folder to view dashboards.

## 5. Trace collection

Deploy an OTLP collector such as the OpenTelemetry operator and set `APP_OTLP_ENDPOINT` in the gateway environment. Install the `observability` dependency group so tracing is enabled. Grafana Tempo or Jaeger can be configured as the backend.

## 6. GPU validation workflow

1. Deploy the GPU InferenceService (`inferenceservice-basic-gpu.yaml`).
2. Submit a sample inference:
   ```bash
   kubectl port-forward svc/polyp-segmentation-transformer 8000:80 -n kserve-inference
   python app/backend/tests/local_smoke_test.py
   ```
3. Check `nvidia-smi` inside the predictor pod to confirm utilization:
   ```bash
   kubectl exec -it deploy/polyp-segmentation-predictor-default -n kserve-inference -- nvidia-smi
   ```
4. Compare metrics in Grafana between CPU and GPU deployments.

## 7. Canary rollouts

To shift more traffic to the canary revision, adjust `canaryTrafficPercent` in `inferenceservice-canary.yaml` and reapply the manifest. Monitor the alerting rules before promoting the GPU version to 100%.

## 8. CI/CD hooks

The Jenkins Helm chart under `Jenkins/` provisions a controller inside the cluster. The Kubeflow pipelines in `training/orchestration/kube_pipeline.py` call Jenkins to update the KServe storage URI upon successful evaluation.

## 9. Required replacements

Replace these tokens before deployment:

| Placeholder | Description |
|-------------|-------------|
| `REPLACE_ME_PROJECT` | GCP project ID for the container registry. |
| `REPLACE_ME_BUCKET` | Cloud storage bucket hosting ONNX models. |
| `REPLACE_ME_REGISTRY` | Container registry URI (GCR / GAR / ECR). |
| `REPLACE_ME_BROKER` | Kafka broker address for response logging. |

Keep this file updated as new environments are added.
