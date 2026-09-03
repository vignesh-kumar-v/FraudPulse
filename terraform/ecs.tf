# Fargate training path.
#
# Why this exists alongside the SageMaker role in main.tf: this AWS account has
# every one of its 142 SageMaker training-instance quotas set to 0, which is the
# default for a new account. CreateTrainingJob fails with
#
#   ResourceLimitExceeded: The account-level service limit 'ml.m5.large for
#   training job usage' is 0 Instances
#
# A quota increase has been requested and is pending AWS review, so the
# SageMaker role and its S3/CloudWatch policy stay in main.tf and
# scripts/sagemaker_train.py works the moment it is granted. Meanwhile Fargate
# needs no quota increase (30 vCPU available by default) and gives the same
# thing the blueprint asked for: managed compute running the training job, with
# the model artifact landing back in S3.
#
# Cost: 0.5 vCPU / 1 GB Fargate is ~$0.025/hr, so a few-minute run is fractions
# of a cent. ECR storage under the lifecycle policy below is negligible.

# --------------------------------------------------------------------------
# Container registry
# --------------------------------------------------------------------------
resource "aws_ecr_repository" "trainer" {
  name                 = "${local.name}-trainer"
  image_tag_mutability = "MUTABLE"
  force_delete         = var.enable_force_destroy

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "trainer" {
  repository = aws_ecr_repository.trainer.name

  # Without this, every push keeps its predecessor forever. Three images is
  # enough to roll back one step and cheap enough to forget about.
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep only the 3 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

# --------------------------------------------------------------------------
# Cluster + logs
# --------------------------------------------------------------------------
resource "aws_ecs_cluster" "training" {
  name = "${local.name}-training"

  setting {
    name  = "containerInsights"
    value = "disabled" # Insights bills per metric; not worth it for batch jobs.
  }
}

resource "aws_cloudwatch_log_group" "ecs_training" {
  name              = "/ecs/${local.name}-trainer"
  retention_in_days = 7
}

# --------------------------------------------------------------------------
# IAM: two roles, deliberately
# --------------------------------------------------------------------------
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

# The EXECUTION role is used by the ECS agent to pull the image and write the
# log stream, before any of your code runs.
resource "aws_iam_role" "ecs_execution" {
  name               = "${local.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  description        = "ECS agent: pull from ECR, write to CloudWatch."
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# The TASK role is what the training code itself gets. Keeping them separate is
# the point: the container never holds ECR pull rights, and the agent never
# holds access to the data.
resource "aws_iam_role" "ecs_task" {
  name               = "${local.name}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  description        = "The training container: read training data, write the model."
}

resource "aws_iam_role_policy" "ecs_task" {
  name = "${local.name}-ecs-task-s3"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadWriteOfflineStoreObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "${aws_s3_bucket.offline_store.arn}/*"
      },
      {
        Sid      = "ListOfflineStore"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.offline_store.arn
      },
    ]
  })
}

# --------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

resource "aws_security_group" "trainer" {
  name        = "${local.name}-trainer"
  description = "Egress-only: the trainer pulls its image and talks to S3."
  vpc_id      = data.aws_vpc.default.id

  # No ingress rule at all. A batch job has nothing to serve, so the correct
  # ingress is none - not 0.0.0.0/0 on a port nothing listens to.
  egress {
    description = "HTTPS to ECR, S3 and CloudWatch"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --------------------------------------------------------------------------
# Task definition
# --------------------------------------------------------------------------
resource "aws_ecs_task_definition" "trainer" {
  family                   = "${local.name}-trainer"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.trainer_cpu
  memory                   = var.trainer_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    # Explicit because the image is built on an arm64 Mac. Leaving this to the
    # default (X86_64) is a guaranteed `exec format error` at task start, and
    # the failure surfaces as a stopped task with no application logs.
    cpu_architecture = var.trainer_cpu_architecture
  }

  container_definitions = jsonencode([
    {
      name      = "trainer"
      image     = "${aws_ecr_repository.trainer.repository_url}:latest"
      essential = true
      environment = [
        { name = "FP_S3_BUCKET", value = aws_s3_bucket.offline_store.id },
        { name = "AWS_DEFAULT_REGION", value = var.aws_region },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_training.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "trainer"
        }
      }
    }
  ])
}
