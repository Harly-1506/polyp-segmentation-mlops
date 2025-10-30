# 💺 Polyp Segmentation System MLOps

## 📚 Overview

**Polyp-Segmentation-System-MLOps** is an end-to-end **medical image segmentation** platform designed to detect and segment **colorectal polyps** from endoscopic images.

This project goes beyond traditional model training — it shows a complete MLOps system that automates the entire machine learning lifecycle: from data ingestion and training pipeline to deployment, monitoring, and continuous delivery.

The platform is orchestrated with **Kubeflow Pipelines** and deployed on **Google Kubernetes Engine (GKE)**, using **KServe (Triton Inference Server)** for scalable model serving.

---

## 🧩 Architecture

## ![System Architecture](docs/architecture.png)

## 🧱 Stack Overview

| Category           | Tools / Frameworks                  |
| ------------------ | ----------------------------------- |
| **Orchestration**  | Kubeflow Pipelines, Jenkins         |
| **Training**       | PyTorch, Ray Train, Ray Tune        |
| **Tracking**       | MLflow (PostgreSQL + MinIO backend) |
| **Deployment**     | KServe, Triton Inference Server     |
| **Storage**        | MinIO, GCS                          |
| **Monitoring**     | Prometheus, Grafana                 |
| **Infrastructure** | GKE, Docker, Terraform              |
| **UI**             | Gradio                              |

---

## 📁 Table of Contents

* [💺 Polyp Segmentation System MLOps](#-polyp-segmentation-system-mlops)

  * [📚 Overview](#-overview)
  * [🧩 Architecture](#-architecture)
  * [🧱 Stack Overview](#-stack-overview)
  * [⚙️ Environment Setup](#%EF%B8%8F-environment-setup)

    * [1. Clone the Repository](#1-clone-the-repository)
    * [2. Create and Activate Environment with `uv`](#2-create-and-activate-environment-with-uv)
  * [🧪 Local Development](#-local-development)

    * [1. Install Docker](#1-install-docker)
    * [2. Run MLflow and MinIO Locally](#2-run-mlflow-and-minio-locally)
    * [3. Launch Local Ray Training](#3-launch-local-ray-training)
  * [☸️ Cluster Setup (Kind / GKE)](#%EF%B8%8F-cluster-setup-kind--gke)

    * [1. Install Helm & Kustomize](#1-install-helm--kustomize)
    * [2. Create Kind Cluster](#2-create-kind-cluster)
  * [📦 Kubeflow Deployment](#-kubeflow-deployment)
  * [🧾 MLflow Deployment](#-mlflow-deployment)
  * [⚡ KubeRay Installation](#-kuberay-installation)
  * [🧽 Namespace & RayCluster Setup](#-namespace--raycluster-setup)
  * [🚀 Pipeline Integration](#-pipeline-integration)
  * [Development & Deployment Playbook (English)](#development--deployment-playbook-english)

    * [1) Local Validation](#1-local-validation)

      * [1.2 Export the model to ONNX](#12-export-the-model-to-onnx)
      * [1.3 Run Triton locally](#13-run-triton-locally)
      * [1.4 Build & run the FastAPI gateway](#14-build--run-the-fastapi-gateway)
    * [2) Build Once, Deploy Anywhere](#2-build-once-deploy-anywhere)

      * [2.1 Authenticate & configure gcloud](#21-authenticate--configure-gcloud)
      * [2.2 Create an Artifact Registry](#22-create-an-artifact-registry)
      * [2.3 Build & push the gateway image](#23-build--push-the-gateway-image)
      * [2.4 Upload model to GCS](#24-upload-model-to-gcs)
    * [3) Deploy to GKE with KServe](#3-deploy-to-gke-with-kserve)

      * [3.1 Provision a cluster](#31-provision-a-cluster)
      * [3.2 Install cert-manager, Istio, Knative Serving, and KServe](#32-install-cert-manager-istio-knative-serving-and-kserve)
      * [3.3 Create Service Accounts (Workload Identity: GSA ↔ KSA)](#33-create-service-accounts-workload-identity-gsa--ksa)
      * [3.4 Deploy the Inference Service](#34-deploy-the-inference-service)
    * [4) Monitoring & Observability](#4-monitoring--observability)
    * [7) Canary Rollouts (Optional)](#7-canary-rollouts-optional)
    * [8) CI/CD Integration (Optional)](#8-cicd-integration-optional)

---



## ⚙️ Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Harly-1506/polyp-segmentation-mlops.git
cd polyp-segmentation-mlops
```

### 2. Create and Activate Environment with `uv`

```bash
conda create -n polyp_mlops python==3.12.9
conda activate polyp_mlops

pip install uv
uv sync --all-groups
```

---

## 🧪 Local Development

You can start with a **local environment** to verify training and MLflow tracking before deploying to the cluster.

### 1. Install Docker

Follow the official Docker installation guide:
👉 [Install Docker Engine](https://docs.docker.com/engine/install)

### 2. Run MLflow and MinIO Locally

```bash
docker compose -f docker-compose-mlflow.yaml up -d --build
```

### 3. Launch Local Ray Training

```bash
uv run --active -m training.ray_main --config training/configs/configs.yaml
```

This runs Ray-based distributed training while logging metrics and artifacts to **MLflow** (using **PostgreSQL** + **MinIO** backends).

---

## ☸️ Cluster Setup (Kind / GKE)

### 1. Install Helm & Kustomize

```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/
```

### 2. Create Kind Cluster

```bash
cat <<EOF | kind create cluster --name=kubeflow --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  image: kindest/node:v1.32.0@sha256:c48c62eac5da28cdadcf560d1d8616cfa6783b58f0d94cf63ad1bf49600cb027
  kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    apiServer:
      extraArgs:
        "service-account-issuer": "https://kubernetes.default.svc"
        "service-account-signing-key-file": "/etc/kubernetes/pki/sa.key"
  extraMounts:
    - hostPath: /home/harly/data
      containerPath: /mnt/data
EOF
```

![Kind Cluster](docs/create_cluster.png)

**Save Kubeconfig**

```bash
kind get kubeconfig --name kubeflow > /tmp/kubeflow-config
export KUBECONFIG=/tmp/kubeflow-config
```

**Create Docker Registry Secret**

```bash
docker login

kubectl create secret generic regcred \
  --from-file=.dockerconfigjson=$HOME/.docker/config.json \
  --type=kubernetes.io/dockerconfigjson
```

---

## 📦 Kubeflow Deployment

Download and deploy the official Kubeflow manifests:

```bash
RELEASE=v1.10.1
git clone -b $RELEASE --depth 1 --single-branch https://github.com/kubeflow/manifests.git
cd manifests

while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do 
  echo "Retrying to apply resources"
  sleep 20
done
```

---

## 🧾 MLflow Deployment

Create a namespace and deploy MLflow via Helm:

```bash
kubectl create namespace mlflow
kubens mlflow

docker build -t harly1506/mlflow-custom:v1.0 .
kind load docker-image harly1506/mlflow-custom:v1.0 --name kubeflow

helm install mlflow ./mlflow -f ./mlflow/values.yaml -n mlflow
helm upgrade mlflow mlflow -f mlflow/values.yaml
```

### Integrate MLflow into Kubeflow Dashboard

Edit the Kubeflow Central Dashboard ConfigMap:

```bash
kubectl -n kubeflow get configmap centraldashboard-config -o yaml
```

Then add this menu item under the dashboard JSON configuration:

```json
{
  "type": "item",
  "link": "/mlflow/",
  "text": "MLflow",
  "icon": "check"
}
```

---

## ⚡ KubeRay Installation

Deploy the KubeRay operator:

```bash
cd ray
kustomize build kuberay-operator/overlays/kubeflow | kubectl apply --server-side -f -
kubectl get pod -l app.kubernetes.io/component=kuberay-operator -n kubeflow
```

Expected output:

```
NAME                                READY   STATUS    RESTARTS   AGE
kuberay-operator-5b8cd69758-rkpvh   1/1     Running   0          6m
```

---

## 🧭 Namespace & RayCluster Setup

Create a new namespace and service account:

```bash
kubectl create ns development
kubectl create sa default-editor -n development
```

Modify the **RayCluster YAML**:

* Update the `AuthorizationPolicy` principal:

  ```yaml
  principals:
  - "cluster.local/ns/development/sa/default-editor"
  ```
* Update the node address for `headGroupSpec` and `workerGroupSpec`:

  ```yaml
  node-ip-address: $(hostname -I | tr -d ' ' | sed 's/\./-/g').raycluster-istio-headless-svc.development.svc.cluster.local
  ```

Deploy RayCluster:

```bash
cd ray
helm install raycluster ray-cluster -n development 
helm upgrade raycluster ray-cluster -n development -f values.yaml
```

---

## 🚀 Pipeline Integration

Build and push the image for Kubeflow Pipeline:

```bash
docker build -t harly1506/polyp-mlops:kfpv2 .
kind load docker-image harly1506/polyp-mlops:kfpv2 --name kubeflow
```

Run this to generate pipeline yaml
```python
 uv run training/orchestration/kube_pipeline.py 
 ``` 

 Upload file ray_segmentation_pipeline_v9.yaml on kubeflow portal
![Training Pipeline](docs/kubeflow-training.png)
 

# Development
End‑to‑end guide to deploy a **Triton Inference Stack** on **GKE with KServe**, with **Prometheus/Grafana** for observability and optional **GPU** + **Canary** + **CI/CD**.

---

## 1) Local Validation

Validate **Triton** and the **FastAPI gateway** before deploying to the cluster.
### 1.2 Export the model to ONNX
```bash
CHECKPOINT="training/checkpoints/UNet/Unet81PolypPVT-best.pth"
ONNX_EXPORT="artifacts/polyp-segmentation/1/model.onnx"

uv run python -m training.scripts.export_to_onnx "${CHECKPOINT}" "${ONNX_EXPORT}" --image-size 256 --dynamic

cat <<'EOF' > "artifacts/polyp-segmentation/config.pbtxt"
name: "polyp-segmentation"
platform: "onnxruntime_onnx"
max_batch_size: 1
input [
  { name: "input", data_type: TYPE_FP32, dims: [3, -1, -1] }
]
output [
  { name: "output", data_type: TYPE_FP32, dims: [1, -1, -1] }
]
instance_group [{ kind: KIND_CPU }]
EOF
```

### 1.3 Run Triton locally
```bash
export MODEL_REPO="$(pwd)/artifacts/polyp-segmentation"

docker run --rm --name triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${MODEL_REPO}:/models/polyp-segmentation" \
  nvcr.io/nvidia/tritonserver:24.10-py3 \
  tritonserver --model-repository=/models \
               --strict-model-config=false \
               --log-verbose=1
```

Health check:
```bash
curl http://localhost:8000/v2/health/ready
```

### 1.4 Build & run the FastAPI gateway
```bash
docker compose -f docker-compose-app.yaml up --build
```


---

## 2) Build and deployment

### 2.1 Authenticate & configure gcloud
```bash
export PROJECT_ID="polyp-mlops-1506"
export REGION="asia-southeast1"
export ARTIFACT_REPO="polyp-inference"

gcloud auth login
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud services enable container.googleapis.com artifactregistry.googleapis.com compute.googleapis.com
```

### 2.2 Create an Artifact Registry (for the FastAPI/Triton gateway image)
```bash
gcloud artifacts repositories create "${ARTIFACT_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Containers for Triton gateway"
```

### 2.3 Build & push the gateway image
```bash
GATEWAY_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/polyp-gateway:latest"

docker build -t "${GATEWAY_IMAGE}" -f app/Dockerfile .
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker push "${GATEWAY_IMAGE}"
```

### 2.4 Upload model to GCS
```bash
MODEL_BUCKET="gs://my-polyp-models"

# (Create bucket if not existing)
gsutil mb -l "${REGION}" "${MODEL_BUCKET}" || echo "Bucket may already exist"

# Upload ONNX + config
gsutil cp "${ONNX_EXPORT}" "${MODEL_BUCKET}/models/polyp-segmentation/1/model.onnx"
gsutil cp artifacts/polyp-segmentation/config.pbtxt "${MODEL_BUCKET}/models/polyp-segmentation/config.pbtxt"
```

---

## 3) Deploy to GKE with KServe

### 3.1 Provision a cluster
If you use Terraform, run:
```bash
terraform init
terraform plan
terraform apply
```

Then connect `kubectl`:
```bash
export CLUSTER="ml-inference-cluster"

# Use --region for regional clusters; use --zone for zonal clusters.
gcloud container clusters get-credentials "${CLUSTER}" --region "${REGION}" --project "${PROJECT_ID}"
```

### 3.2 Install cert-manager, Istio, Knative Serving, and KServe
```bash
# 0) cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.19.1/cert-manager.yaml
kubectl -n cert-manager wait --for=condition=Available deploy --all --timeout=300s

# 1) Istio (control plane + ingressgateway)
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.22.3 sh -
./istio-1.22.3/bin/istioctl install -y --set profile=default
kubectl -n istio-system wait --for=condition=Available deploy --all --timeout=300s
kubectl -n istio-system get svc istio-ingressgateway

# 2) Knative Serving (CRDs + core)
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.19.5/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.19.5/serving-core.yaml
kubectl -n knative-serving wait --for=condition=Available deploy --all --timeout=300s

# 3) Knative net-istio (use Istio as ingress)
kubectl apply -f https://github.com/knative-extensions/net-istio/releases/download/knative-v1.19.5/net-istio.yaml
kubectl -n knative-serving patch configmap/config-network \
  --type merge -p='{"data":{"ingress.class":"istio.ingress.networking.knative.dev"}}'

# 4) KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.15.2/kserve.yaml --server-side --force-conflicts
kubectl -n kserve wait --for=condition=Available deploy/kserve-controller-manager --timeout=300s

kubectl apply -f "https://github.com/kserve/kserve/releases/download/v0.15.2/kserve-cluster-resources.yaml"
# 5) Verify webhook is ready
kubectl -n kserve get endpoints kserve-webhook-server-service
```

![Grafana](docs/installed-kserve.png)

### 3.3 Create Service Accounts (Workload Identity: GSA ↔ KSA)

**Goal:** allow KServe Pods to read the model from your **GCS bucket** without any key files, using **Workload Identity**.

```bash
export NAMESPACE="kserve-inference"
export BUCKET="my-polyp-models"              
export GSA="kserve-infer-sa"                
export KSA="kserve-model-sa"                 

kubectl create namespace "${NAMESPACE}" || true

# Enable Workload Identity on the cluster
gcloud container clusters update "${CLUSTER}" --region "${REGION}" \
  --workload-pool="${PROJECT_ID}.svc.id.goog"

# Create a Google Service Account (GSA)
gcloud iam service-accounts create "${GSA}" --project "${PROJECT_ID}"

# Grant minimal read access to the model bucket (object viewer)
gsutil iam ch serviceAccount:${GSA}@${PROJECT_ID}.iam.gserviceaccount.com:roles/storage.objectViewer gs://${BUCKET}

# Create a Kubernetes Service Account (KSA)
kubectl -n "${NAMESPACE}" create serviceaccount "${KSA}" || true

# Bind GSA ↔ KSA (Workload Identity)
gcloud iam service-accounts add-iam-policy-binding \
  "${GSA}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/iam.workloadIdentityUser" \
  --member "serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/${KSA}]"

# Annotate KSA with the GSA identity
kubectl -n "${NAMESPACE}" annotate serviceaccount "${KSA}" \
  iam.gke.io/gcp-service-account="${GSA}@${PROJECT_ID}.iam.gserviceaccount.com" --overwrite
```

**Short explanation:**  
- **GSA** (Google Service Account): holds IAM permissions on GCP resources (GCS bucket).  
- **KSA** (Kubernetes Service Account): attached to your Pods.  
- **Workload Identity** links KSA → GSA, so Pods transparently use the GSA’s permissions—no JSON key files needed.

> **Tip:** Ensure your InferenceService pods run with `serviceAccountName: kserve-model-sa` (or the `${KSA}` you created).

### 3.4 Deploy the Inference Service
```bash
kubectl apply -k deployment/kserve
kubectl apply -k deployment/ui
```
### 3.5 Open UI:
```bash
k port-forward svc/polyp-ui 7860:80 -n kserve-inference 
```
-  Get external IP with Nginx
```bash

kubectl -n kserve-inference get svc polyp-ui-nginx
```
Output should be: 
```
NAME             TYPE           CLUSTER-IP   EXTERNAL-IP      PORT(S)        AGE
polyp-ui-nginx   LoadBalancer   10.30.1.4    34.124.246.185   80:31912/TCP   9d
```

![UI](docs/UI.png)

## 4) Monitoring & Observability
Install Prometheus + Grafana (Helm).
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts

helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace observability --create-namespace \
  -f deployment/monitoring/prometheus/values.yaml

helm upgrade --install grafana grafana/grafana \
  --namespace observability \
  -f deployment/monitoring/grafana/values.yaml

kubectl apply -k deployment/monitoring
```

Access to grafana and Prometheus:
```bash
kubectl port-forward svc/prometheus-operated 9090:9090 -n observability 
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n observability 
```
![Grafana](docs/Grafana.png)
![Grafana](docs/Grafana1.png)
![Grafana](docs/Grafana2.png)
---
![Grafana](docs/Prometheus.png)

---

## 7) Canary Rollouts (Optional)
```bash
kubectl apply -f deployment/kserve/inferenceservice-canary.yaml -n "${NAMESPACE}"
```
Adjust `canaryTrafficPercent` to shift traffic between the old and new model revisions.

---

## 8) CI/CD Integration (Optional) Updating
- Use Jenkins (chart under `Jenkins/`) to build, push, and update KServe manifests.
- In **Kubeflow Pipelines** (`training/orchestration/kube_pipeline.py`), update `storageUri` automatically when a model passes evaluation.
- Optionally trigger Jenkins to upload the best checkpoint to GCS upon promotion.

---

