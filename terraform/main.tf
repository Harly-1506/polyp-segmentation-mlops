terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "required" {
  for_each = toset(var.enabled_services)

  project = var.project_id
  service = each.value
}

resource "google_compute_network" "inference" {
  name                    = var.vpc_network_name
  auto_create_subnetworks = false

  depends_on = [google_project_service.required]
}

resource "google_compute_subnetwork" "inference" {
  name          = var.subnet_name
  ip_cidr_range = var.subnet_cidr
  network       = google_compute_network.inference.id
  region        = var.region

  secondary_ip_range {
    range_name    = var.pods_range_name
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = var.services_range_name
    ip_cidr_range = var.services_cidr
  }

  depends_on = [
    google_project_service.required,
    google_compute_network.inference
  ]
}

resource "google_container_cluster" "inference" {
  name     = var.cluster_name
  location = var.region
  project  = var.project_id

  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection = false
  node_config {
    disk_type    = var.primary_disk_type
    disk_size_gb = var.primary_disk_size_gb
  }

  release_channel {
    channel = var.release_channel
  }

  networking_mode = "VPC_NATIVE"
  network         = google_compute_network.inference.self_link
  subnetwork      = google_compute_subnetwork.inference.self_link

  ip_allocation_policy {
    cluster_secondary_range_name  = var.pods_range_name
    services_secondary_range_name = var.services_range_name
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  depends_on = [google_project_service.required]
}

resource "google_container_node_pool" "primary" {
  name     = "primary-pool"
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.inference.name

  node_config {
    machine_type    = var.primary_machine_type
    service_account = var.primary_service_account != "" ? var.primary_service_account : null
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    disk_type       = var.primary_disk_type
    disk_size_gb    = var.primary_disk_size_gb
    image_type      = var.primary_image_type
    metadata        = var.primary_node_metadata
    labels          = var.primary_node_labels
    tags            = var.primary_node_tags
  }

  autoscaling {
    min_node_count = var.primary_min_nodes
    max_node_count = var.primary_max_nodes
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  depends_on = [google_container_cluster.inference]
}

resource "google_container_node_pool" "spot_pool" {
  count    = var.enable_spot_pool ? 1 : 0
  name     = "spot-pool"
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.inference.name

  node_config {
    spot            = true
    machine_type    = var.spot_machine_type
    service_account = var.spot_service_account != "" ? var.spot_service_account : null
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    disk_type       = var.spot_disk_type
    disk_size_gb    = var.spot_disk_size_gb
    image_type      = var.spot_image_type
    metadata        = var.spot_node_metadata
    labels          = var.spot_node_labels
    tags            = var.spot_node_tags
  }

  autoscaling {
    min_node_count = var.spot_min_nodes
    max_node_count = var.spot_max_nodes
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  depends_on = [google_container_node_pool.primary]
}

output "cluster_name" {
  value       = google_container_cluster.inference.name
  description = "Name of the created GKE cluster."
}

output "cluster_endpoint" {
  value       = google_container_cluster.inference.endpoint
  description = "Endpoint of the created GKE cluster."
}

output "cluster_ca_certificate" {
  value       = google_container_cluster.inference.master_auth[0].cluster_ca_certificate
  description = "Base64 encoded public certificate authority for the cluster."
}
