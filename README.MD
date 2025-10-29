# 🩺 Polyp Segmentation System MLOps

## 📘 Overview

**Polyp-Segmentation-System-MLOps** is an end-to-end **medical image segmentation** platform designed to detect and segment **colorectal polyps** from endoscopic images.

This project goes beyond traditional model training — it show a complete MLOps system that automates the entire machine learning lifecycle: from data ingestion and training pipeline to deployment, monitoring, and continuous delivery.

The platform is orchestrated with **Kubeflow Pipelines** and deployed on **Google Kubernetes Engine (GKE)**, using **KServe (Triton Inference Server)** for scalable model serving.  

---

## 🧩 Architecture 
![System Architecture](docs/architecture.png)
---

## 🧱 Stack Overview

| Category | Tools / Frameworks |
|-----------|-------------------|
| **Orchestration** | Kubeflow Pipelines, Jenkins |
| **Training** | PyTorch, Ray Train, Ray Tune |
| **Tracking** | MLflow (PostgreSQL + MinIO backend) |
| **Deployment** | KServe, Triton Inference Server |
| **Storage** | MinIO, GCS |
| **Monitoring** | Prometheus, Grafana |
| **Infrastructure** | GKE, Docker, Terraform |
| **UI** | Gradio  |


## Table of content



## Set up the environment

1. Clone the repository
   ```bash
   git clone https://github.com/Harly-1506/polyp-segmentation-mlops.git
   cd polyp-segmentation-mlops
   ```
2. Install `uv` and Synchronize dependencies:
   ```bash
   conda create -n polyp_mlops python==3.12.9                                                             
   conda activate polyp_mlops
   pip install uv
   uv sync --frozen --group ml --group inference --group observability
   ```
