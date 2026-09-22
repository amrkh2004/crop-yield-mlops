output "s3_bucket_name" {
  value       = aws_s3_bucket.dvc_remote.id
  description = "The name of the created S3 Bucket"
}

output "s3_bucket_arn" {
  value       = aws_s3_bucket.dvc_remote.arn
  description = "The ARN of the created S3 Bucket"
}

output "dvc_remote_url" {
  value       = "s3://${aws_s3_bucket.dvc_remote.id}/dvc-store"
  description = "The target DVC Remote URL for dvc remote add command"
}
