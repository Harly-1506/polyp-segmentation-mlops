# Copy this file to terraform.tfvars and update the values for your environment.
project_id           = "polyp-mlops-1506"
region               = "asia-southeast1"
cluster_name         = "ml-inference-cluster"
vpc_network_name     = "inference-vpc-network"
subnet_name          = "inference-subnet"
subnet_cidr          = "10.10.0.0/24"
pods_range_name      = "pods"
pods_cidr            = "10.20.0.0/16"
services_range_name  = "services"
services_cidr        = "10.30.0.0/20"
primary_machine_type = "e2-standard-4"
primary_min_nodes    = 1
primary_max_nodes    = 2

# Optional spot node pool example configuration
enable_spot_pool  = false
spot_machine_type = "e2-standard-4"
spot_min_nodes    = 1
spot_max_nodes    = 3
