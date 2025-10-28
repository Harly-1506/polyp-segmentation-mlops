variable "project_id" {
  description = "GCP project ID where the resources will be created."
  type        = string
}

variable "region" {
  description = "GCP region for the GKE cluster."
  type        = string
  default     = "asia-southeast1"
}

variable "enabled_services" {
  description = "List of Google APIs that must be enabled for the deployment."
  type        = list(string)
  default = [
    "compute.googleapis.com",
    "container.googleapis.com",
    "iam.googleapis.com",
    "artifactregistry.googleapis.com"
  ]
}

variable "vpc_network_name" {
  description = "Name of the existing VPC network the cluster should attach to."
  type        = string
  default     = "inference-vpc"
}

variable "subnet_name" {
  description = "Name of the subnetwork to create for the cluster."
  type        = string
  default     = "inference-subnet"
}

variable "subnet_cidr" {
  description = "Primary IPv4 CIDR block for the managed subnetwork."
  type        = string
  default     = "10.10.0.0/24"
}

variable "pods_range_name" {
  description = "Name of the secondary IP range for Kubernetes pods."
  type        = string
  default     = "pods"
}

variable "pods_cidr" {
  description = "IPv4 CIDR block for the pod secondary range."
  type        = string
  default     = "10.20.0.0/16"
}

variable "services_range_name" {
  description = "Name of the secondary IP range for Kubernetes services."
  type        = string
  default     = "services"
}

variable "services_cidr" {
  description = "IPv4 CIDR block for the services secondary range."
  type        = string
  default     = "10.30.0.0/20"
}

variable "cluster_name" {
  description = "Name of the GKE cluster."
  type        = string
  default     = "inference-cluster"
}

variable "release_channel" {
  description = "Release channel for the GKE cluster (e.g., RAPID, REGULAR, STABLE)."
  type        = string
  default     = "REGULAR"
}

variable "primary_machine_type" {
  description = "Machine type for the primary node pool."
  type        = string
  default     = "e2-standard-2"
}

variable "primary_min_nodes" {
  description = "Minimum number of nodes for the primary node pool."
  type        = number
  default     = 1
}

variable "primary_max_nodes" {
  description = "Maximum number of nodes for the primary node pool."
  type        = number
  default     = 1
}

variable "primary_disk_type" {
  description = "Disk type for nodes in the primary node pool."
  type        = string
  default     = "pd-standard"
}

variable "primary_disk_size_gb" {
  description = "Disk size in GB for nodes in the primary node pool."
  type        = number
  default     = 100
}

variable "primary_image_type" {
  description = "Node image type for the primary node pool."
  type        = string
  default     = "COS_CONTAINERD"
}

variable "primary_service_account" {
  description = "Service account email for the primary node pool. Leave empty to use the default compute service account."
  type        = string
  default     = ""
}

variable "primary_node_metadata" {
  description = "Metadata to assign to nodes in the primary node pool."
  type        = map(string)
  default     = {}
}

variable "primary_node_labels" {
  description = "Kubernetes labels to assign to nodes in the primary node pool."
  type        = map(string)
  default     = {}
}

variable "primary_node_tags" {
  description = "Network tags to assign to nodes in the primary node pool."
  type        = list(string)
  default     = []
}

variable "enable_spot_pool" {
  description = "Whether to provision an additional spot node pool for cost savings."
  type        = bool
  default     = false
}

variable "spot_machine_type" {
  description = "Machine type for the optional spot node pool."
  type        = string
  default     = "e2-standard-2"
}

variable "spot_min_nodes" {
  description = "Minimum number of nodes for the spot node pool."
  type        = number
  default     = 0
}

variable "spot_max_nodes" {
  description = "Maximum number of nodes for the spot node pool."
  type        = number
  default     = 3
}

variable "spot_disk_type" {
  description = "Disk type for nodes in the optional spot pool."
  type        = string
  default     = "pd-standard"
}

variable "spot_disk_size_gb" {
  description = "Disk size in GB for nodes in the optional spot pool."
  type        = number
  default     = 50
}

variable "spot_image_type" {
  description = "Node image type for the optional spot pool."
  type        = string
  default     = "COS_CONTAINERD"
}

variable "spot_service_account" {
  description = "Service account email for the optional spot pool. Leave empty to use the default compute service account."
  type        = string
  default     = ""
}

variable "spot_node_metadata" {
  description = "Metadata to assign to nodes in the optional spot pool."
  type        = map(string)
  default     = {}
}

variable "spot_node_labels" {
  description = "Kubernetes labels to assign to nodes in the optional spot pool."
  type        = map(string)
  default     = {}
}

variable "spot_node_tags" {
  description = "Network tags to assign to nodes in the optional spot pool."
  type        = list(string)
  default     = []
}
