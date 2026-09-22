variable "aws_region" {
  type        = string
  description = "AWS region for provisioning infrastructure"
  default     = "us-east-1"
}

variable "bucket_name" {
  type        = string
  description = "Unique AWS S3 bucket name for DVC Remote Storage"
  default     = "crop-yield-dvc-remote-store"
}

variable "environment" {
  type        = string
  description = "Deployment environment name (e.g. development, staging, production)"
  default     = "production"
}

variable "force_destroy" {
  type        = bool
  description = "Whether to force destroy all objects when deleting bucket"
  default     = false
}
