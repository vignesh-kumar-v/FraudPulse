variable "aws_region" {
  description = "Region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment tag; also part of every resource name."
  type        = string
  default     = "dev"

  validation {
    condition     = can(regex("^[a-z0-9-]{2,12}$", var.environment))
    error_message = "environment must be 2-12 lowercase alphanumeric/hyphen characters."
  }
}

variable "project" {
  description = "Name prefix for every resource."
  type        = string
  default     = "fraudpulse"
}

variable "offline_store_retention_days" {
  description = <<-EOT
    Days before offline feature Parquet expires. Set deliberately low: this is a
    portfolio project and an S3 bucket that quietly accumulates feature
    snapshots is the classic way a "free tier" stops being free.
  EOT
  type        = number
  default     = 14

  validation {
    condition     = var.offline_store_retention_days >= 1
    error_message = "retention must be at least 1 day."
  }
}

variable "training_instance_type" {
  description = <<-EOT
    SageMaker training instance. ml.m5.large is ~$0.115/hr; the FraudPulse
    training job runs in a few minutes, so a full run costs roughly one cent.
    Anything larger buys nothing on a 590k x 31 dataset.
  EOT
  type        = string
  default     = "ml.m5.large"
}

variable "enable_force_destroy" {
  description = <<-EOT
    Let `terraform destroy` delete a non-empty offline-store bucket.

    true here on purpose, and it is a real tradeoff: it is what makes teardown
    reliably clean (S3 refuses to delete a bucket with objects in it, which is
    the single most common cause of a half-finished destroy and an orphaned
    bucket still costing money). It also means destroy silently deletes data.
    Correct for a project bucket that is rebuilt from local Parquet in one
    command; wrong for anything you cannot regenerate.
  EOT
  type        = bool
  default     = true
}

variable "trainer_cpu" {
  description = "Fargate CPU units. 512 = 0.5 vCPU (~$0.025/hr with 1GB)."
  type        = string
  default     = "1024"
}

variable "trainer_memory" {
  description = "Fargate memory in MiB. Must be a valid pairing with trainer_cpu."
  type        = string
  default     = "4096"
}

variable "trainer_cpu_architecture" {
  description = <<-EOT
    ARM64 or X86_64. Must match the architecture the image was actually built
    for; a mismatch fails at container start with `exec format error` and no
    application logs, which is a genuinely annoying thing to debug. ARM64 is
    also ~20% cheaper on Fargate and is what an Apple Silicon `docker build`
    produces by default.
  EOT
  type        = string
  default     = "ARM64"

  validation {
    condition     = contains(["ARM64", "X86_64"], var.trainer_cpu_architecture)
    error_message = "must be ARM64 or X86_64."
  }
}
