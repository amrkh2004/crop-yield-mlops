terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_s3_bucket" "dvc_remote" {
  bucket        = var.bucket_name
  force_destroy = var.force_destroy

  tags = {
    Name        = var.bucket_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Purpose     = "DVC Remote Storage for Crop Yield MLOps"
  }
}

resource "aws_s3_bucket_versioning" "dvc_remote_versioning" {
  bucket = aws_s3_bucket.dvc_remote.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dvc_remote_encryption" {
  bucket = aws_s3_bucket.dvc_remote.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "dvc_remote_public_block" {
  bucket = aws_s3_bucket.dvc_remote.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
