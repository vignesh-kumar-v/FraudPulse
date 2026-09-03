# FraudPulse AWS footprint.
#
# What this provisions, and what it deliberately does not:
#
#   DOES  S3 bucket for the Feast offline store (versioned, encrypted, private,
#         lifecycle-expired), an IAM execution role SageMaker can assume with
#         least-privilege access to exactly that bucket, and a CloudWatch log
#         group for training output.
#
#   DOES NOT  create a SageMaker *endpoint*. An endpoint is a running instance
#         billed by the hour whether or not anything calls it, and leaving one
#         up is the most expensive mistake available in this API. Training jobs
#         are billed per second of actual training and are submitted by
#         scripts/sagemaker_train.py against the role below.
#
# Every resource carries the default_tags from versions.tf, so a failed destroy
# can be found with one tag query rather than by clicking through the console.

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Bucket names are globally unique across all of AWS, so a fixed name would
# collide with anyone else who ran this. The suffix is stored in state, which
# keeps it stable across applies.
resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  name        = "${var.project}-${var.environment}"
  bucket_name = "${local.name}-offline-store-${random_id.suffix.hex}"
  account_id  = data.aws_caller_identity.current.account_id
}

# --------------------------------------------------------------------------
# Offline feature store
# --------------------------------------------------------------------------
resource "aws_s3_bucket" "offline_store" {
  bucket = local.bucket_name

  # See the variable docs: this is what makes destroy reliable, at the cost of
  # destroy being able to delete data.
  force_destroy = var.enable_force_destroy
}

resource "aws_s3_bucket_versioning" "offline_store" {
  bucket = aws_s3_bucket.offline_store.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "offline_store" {
  bucket = aws_s3_bucket.offline_store.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "offline_store" {
  bucket                  = aws_s3_bucket.offline_store.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "offline_store" {
  bucket     = aws_s3_bucket.offline_store.id
  depends_on = [aws_s3_bucket_versioning.offline_store]

  rule {
    id     = "expire-feature-snapshots"
    status = "Enabled"
    filter {}

    expiration {
      days = var.offline_store_retention_days
    }

    # Versioning is on, so expiring the current version only hides objects
    # behind delete markers - the noncurrent versions keep billing. Both rules
    # are needed for the retention setting to actually mean anything.
    noncurrent_version_expiration {
      noncurrent_days = 1
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# --------------------------------------------------------------------------
# SageMaker execution role
# --------------------------------------------------------------------------
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }

    # Confused-deputy guard: without this, any SageMaker job in any account
    # that somehow references this role ARN could assume it.
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

data "aws_iam_policy_document" "training" {
  # Read the training data, write the model artifact back. Scoped to this one
  # bucket rather than the AmazonSageMakerFullAccess managed policy, which
  # grants s3:* on every bucket whose name contains "sagemaker".
  statement {
    sid       = "OfflineStoreObjects"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.offline_store.arn}/*"]
  }

  statement {
    sid       = "OfflineStoreList"
    effect    = "Allow"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.offline_store.arn]
  }

  statement {
    sid    = "TrainingLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = ["${aws_cloudwatch_log_group.training.arn}:*"]
  }

  statement {
    sid       = "TrainingMetrics"
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    # cloudwatch:PutMetricData has no resource-level permissions, so "*" is
    # unavoidable. The namespace condition is what keeps it from being a
    # write-anywhere grant.
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/TrainingJobs"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${local.name}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
  description        = "Execution role for FraudPulse SageMaker training jobs."
}

resource "aws_iam_role_policy" "training" {
  name   = "${local.name}-training"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.training.json
}

# --------------------------------------------------------------------------
# Training logs
# --------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "training" {
  name = "/aws/sagemaker/TrainingJobs/${local.name}"

  # SageMaker's own log group defaults to never expiring. A log group with no
  # retention is a small bill that grows forever and survives `terraform
  # destroy` if it was created by the service instead of by this config -
  # which is exactly why it is declared here.
  retention_in_days = 7
}
