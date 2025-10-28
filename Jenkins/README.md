# Jenkins for Automated Triton Deployments

This Helm chart provisions a lightweight Jenkins controller dedicated to promoting new Triton models in KServe. Configuration-as-Code bootstraps credentials and a pipeline job that patches the `InferenceService` once a Kubeflow pipeline signals success.

## Installing

```bash
helm upgrade --install polyp-jenkins ./Jenkins --namespace jenkins --create-namespace
```

> **Important:** Update `values.yaml` before installation:
>
> - Provide administrator credentials either by setting `controller.adminPassword` (Helm will generate a random password when le
ft blank) or by referencing a pre-created Kubernetes secret via `controller.existingAdminSecret`. For production clusters, crea
te the secret yourself, for example:
>
>   ```bash
>   kubectl create secret generic jenkins-admin --from-literal=password="$(openssl rand -base64 32)" -n jenkins
>   helm upgrade --install polyp-jenkins ./Jenkins \
>     --namespace jenkins --create-namespace \
>     --set controller.existingAdminSecret=jenkins-admin \
>     --set controller.adminUser=jenkins
>   ```
>
> - Optionally enable ingress and TLS for production.
> - Adjust the storage class to match your cluster.

## Pre-requisites

1. Jenkins requires cluster-admin permissions to patch the KServe `InferenceService`. The provided `ClusterRoleBinding` grants this access.
2. Install `kubectl` inside the Jenkins container or mount it as a sidecar. The default image already contains `kubectl`.
3. Provide a Kubernetes service account token for in-cluster authentication (handled automatically when the service account is mounted).

## Networking (Istio vs. standard ingress)

Jenkins does **not** require Istio to run inside the cluster—the chart exposes a `ClusterIP` service that can be reached by
in-cluster workloads (for example, Kubeflow pipeline steps calling the build API). You only need Istio if your platform mandates
all HTTP traffic go through the mesh or if you want to reuse the existing Istio ingress gateway for external access.

- **Without Istio:** enable the optional Kubernetes ingress in `values.yaml` or switch the service type to `LoadBalancer` to
  expose Jenkins.
- **With Istio:** keep the service as `ClusterIP` and create a `Gateway`/`VirtualService` that routes to `polyp-jenkins` on port
  8080. No additional chart changes are required.

Regardless of the approach, ensure that whichever endpoint you expose is secured (TLS + authentication) before allowing
external access.

## Pipeline Contract

The job `polyp-canary-promotion` expects the following parameters, typically forwarded from the Kubeflow pipeline step:

- `MODEL_URI` – fully-qualified GCS/S3 URI for the exported ONNX model (`gs://bucket/path/model.onnx`).
- `CANARY_PERCENT` – optional integer for traffic split (defaults to 25).

The pipeline performs:

1. Patch the canary `storageUri`.
2. Adjust canary traffic weight.
3. Wait for the transformer deployment and InferenceService readiness.

If the health gate fails, Jenkins exits non-zero; your pipeline should catch this and trigger a rollback (set `CANARY_PERCENT=0`).

## Integrating with Kubeflow

The Kubeflow pipelines live in `training/orchestration/kube_pipeline.py`. A typical training run invokes `ray_segmentation_training_pipeline`, which prepares the dataset, executes distributed Ray training, evaluates the checkpoint, and publishes the metrics artifact. When accuracy gates pass, the deployment pipeline `segmentation_deploy_pipeline` exports the winning checkpoint to ONNX and calls Jenkins via the `trigger_jenkins_component`.

To connect Kubeflow to Jenkins:

1. Create a Kubeflow secret (for example `jenkins-token`) that stores the Jenkins API token under the `token` key.
2. Configure the pipeline run with the Jenkins base URL and user—by default `segmentation_deploy_pipeline` expects `http://polyp-jenkins.jenkins:8080` and the `admin` user, but these can be overridden via pipeline parameters.
3. Ensure the training pipeline passes the checkpoint URI (GCS/S3 path), desired ONNX export path, and environment metadata (`target_env`, `model_name`, `model_version`) to the deployment pipeline. The generated ONNX artifact is forwarded to the Jenkins promotion job so traffic can be shifted in KServe once the build succeeds.

The sample pipeline definitions in the repository demonstrate how to wire these pieces together, so recompiling with `python training/orchestration/kube_pipeline.py --pipeline deploy --output pipeline.yaml` produces a manifest that already includes the Jenkins trigger step. Supply the Jenkins URL and API token via Kubeflow runtime parameters or secrets.

## Tear down

```bash
helm uninstall polyp-jenkins -n jenkins
kubectl delete pvc polyp-jenkins -n jenkins
```

The persistent volume claim keeps Jenkins state (credentials, job history) between upgrades. Delete only when you want a fresh instance.