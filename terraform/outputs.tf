output "offline_store_bucket" {
  description = "S3 bucket backing the Feast offline store."
  value       = aws_s3_bucket.offline_store.id
}

output "offline_store_uri" {
  description = "s3:// prefix for feature Parquet."
  value       = "s3://${aws_s3_bucket.offline_store.id}/feature_repo/"
}

output "training_input_uri" {
  description = "Where scripts/sagemaker_train.py uploads the training CSV."
  value       = "s3://${aws_s3_bucket.offline_store.id}/training/input/"
}

output "training_output_uri" {
  description = "Where SageMaker writes model.tar.gz."
  value       = "s3://${aws_s3_bucket.offline_store.id}/training/output/"
}

output "sagemaker_role_arn" {
  description = "Role a SageMaker training job assumes."
  value       = aws_iam_role.sagemaker_execution.arn
}

output "log_group" {
  description = "CloudWatch log group for training jobs."
  value       = aws_cloudwatch_log_group.training.name
}

output "region" {
  value = data.aws_region.current.name
}

output "training_instance_type" {
  value = var.training_instance_type
}

output "ecr_repository_url" {
  description = "Push the trainer image here."
  value       = aws_ecr_repository.trainer.repository_url
}

output "ecs_cluster" {
  value = aws_ecs_cluster.training.name
}

output "ecs_task_definition" {
  value = aws_ecs_task_definition.trainer.family
}

output "ecs_subnets" {
  value = data.aws_subnets.default.ids
}

output "ecs_security_group" {
  value = aws_security_group.trainer.id
}

output "ecs_log_group" {
  value = aws_cloudwatch_log_group.ecs_training.name
}

output "sagemaker_quota_note" {
  description = "Why the Fargate path exists."
  value = join(" ", [
    "SageMaker training-instance quotas are 0 on a new AWS account;",
    "an increase for ml.m5.large (L-611FA074) has been requested.",
    "Until it is granted, ECS Fargate runs the training job.",
  ])
}
