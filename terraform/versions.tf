terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # State is local on purpose.
  #
  # An S3 remote backend would be the right answer for a team, but it is a
  # bootstrapping paradox in a single-module repo: the bucket that holds the
  # state has to exist before the config that creates buckets can run. The
  # usual resolution is a separate bootstrap module applied once by hand.
  # Documented rather than half-built - see terraform/README.md.
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "fraudpulse"
      ManagedBy   = "terraform"
      Environment = var.environment
      # Makes orphaned resources findable with a single tag query if a destroy
      # ever fails partway. See the teardown check in terraform/README.md.
      Repo = "github.com/vignesh-kumar-v/FraudPulse"
    }
  }
}
