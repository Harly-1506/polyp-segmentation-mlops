# Terraform GKE Inference Infrastructure

This Terraform configuration provisions the Google Kubernetes Engine (GKE) infrastructure needed to run inference workloads. It creates a dedicated VPC with secondary ranges, provisions a minimal-cost primary node pool (with an optional spot pool), and exposes outputs that make it easy to connect with `kubectl` or GitOps tooling. Application deployments are intentionally left outside of Terraform so you can manage them independently.

## Files

- `main.tf` – Providers, networking, GKE cluster, and node pools.
- `variables.tf` – Input variables with descriptions and defaults.
- `terraform.tfvars.example` – Sample variable file to copy and adapt to your environment.

## Prerequisites

1. [Install Terraform](https://developer.hashicorp.com/terraform/downloads) v1.5.0 or newer.
2. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and authenticate: `gcloud auth application-default login`.
3. Ensure the Google Cloud project has billing enabled.
4. Verify that the authenticated identity has permission to manage GKE and attach to the target network resources. For long-lived or
   automated usage (for example from Jenkins) create a dedicated Google Cloud service account and grant it at least:

   - `roles/container.admin` – manage the GKE cluster and node pools.
   - `roles/compute.networkAdmin` – create and manage the VPC and subnet resources.
   - `roles/iam.serviceAccountUser` – allow the Terraform-managed GKE node pools to impersonate their node service accounts.

   Then download a JSON key for local Terraform runs or, preferably, bind the service account to your CI/CD workload identity
   (Workload Identity for GKE, Workload Identity Federation, etc.) so you can avoid storing long-lived secrets.

## Usage

1. Copy the example variable file and update the values:

   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Review and adjust the variables in `terraform.tfvars`. The most important ones are listed below.
3. Initialize and apply the configuration:

   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

   Confirm the plan to create the resources.

## Key Variables to Review

| Variable | Description | When to change |
| --- | --- | --- |
| `project_id` | Target Google Cloud project ID. | Always update to your project. |
| `region` | Region for the cluster and subnetwork. | Change to the region closest to your users. |
| `cluster_name` | Name of the GKE cluster. | Optional rename to match naming standards. |
| `primary_machine_type`, `primary_min_nodes`, `primary_max_nodes` | Shape and size of the main node pool. Defaults aim for the lowest cost while keeping one node available. | Adjust to fit workload requirements. |
| `vpc_network_name`, `subnet_name`, `subnet_cidr` | Naming and IP range for the dedicated VPC/subnet created for the cluster. | Adjust to fit your IP plan. |
| `pods_range_name`, `pods_cidr`, `services_range_name`, `services_cidr` | Secondary IP ranges for pods and services. | Ensure they do not overlap with other networks. |
| `enable_spot_pool` and related `spot_*` vars | Optional spot/preemptible node pool configuration. | Enable for cost optimization if workload tolerates disruption. |

The configuration provisions a new VPC-native subnet with the specified secondary ranges. For the full list of variables, refer to `variables.tf`.

## Outputs

- `cluster_name` – Name of the created GKE cluster.
- `cluster_endpoint` – API endpoint for the cluster.
- `cluster_ca_certificate` – Base64 encoded public certificate authority for the cluster. Decode it when building a kubeconfig entry.

## Deploying applications

Terraform now stops at provisioning the cluster. Deploy your applications using Kubernetes manifests or Helm charts once you connect with `kubectl`:

```bash
gcloud container clusters get-credentials <cluster_name> --region <region> --project <project_id>
kubectl apply -f ../deployment/
```

The repository's [`deployment/`](../deployment) directory contains reference manifests (e.g., KServe, monitoring, UI) you can reuse or adapt. Alternatively, manage workloads with your preferred GitOps or CI/CD tooling.

## Cleanup

Destroy resources when no longer needed to avoid charges:

```bash
terraform destroy
```