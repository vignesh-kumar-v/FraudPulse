#!/usr/bin/env bash
# Phase 6 gate: prove `terraform destroy` left nothing behind.
#
# `Destroy complete!` only means Terraform removed everything it had in state.
# Anything created outside Terraform - a log group the service auto-created, an
# ECR image pushed by CI, a bucket a failed apply never recorded - survives it
# silently and keeps billing. This queries AWS directly instead.
set -uo pipefail
REGION="${AWS_REGION:-us-east-1}"
PREFIX="${1:-fraudpulse}"
FOUND=0

check() {
  local label="$1"; shift
  local out
  out="$("$@" 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "$out" && "$out" != "None" ]]; then
    echo "  ORPHAN  $label: $out"
    FOUND=$((FOUND + 1))
  else
    echo "  clean   $label"
  fi
}

echo "scanning region=$REGION prefix=$PREFIX"

check "s3 buckets" aws s3api list-buckets \
  --query "Buckets[?starts_with(Name,'${PREFIX}')].Name" --output text
check "iam roles" aws iam list-roles \
  --query "Roles[?starts_with(RoleName,'${PREFIX}')].RoleName" --output text
check "ecr repositories" aws ecr describe-repositories --region "$REGION" \
  --query "repositories[?starts_with(repositoryName,'${PREFIX}')].repositoryName" --output text
check "ecs clusters" aws ecs list-clusters --region "$REGION" \
  --query "clusterArns[?contains(@,'${PREFIX}')]" --output text
check "ecs task definitions" aws ecs list-task-definitions --region "$REGION" --status ACTIVE \
  --query "taskDefinitionArns[?contains(@,'${PREFIX}')]" --output text
check "cloudwatch log groups" aws logs describe-log-groups --region "$REGION" \
  --query "logGroups[?contains(logGroupName,'${PREFIX}')].logGroupName" --output text
check "security groups" aws ec2 describe-security-groups --region "$REGION" \
  --query "SecurityGroups[?starts_with(GroupName,'${PREFIX}')].GroupName" --output text
check "sagemaker training jobs (in progress)" aws sagemaker list-training-jobs --region "$REGION" \
  --status-equals InProgress --query "TrainingJobSummaries[].TrainingJobName" --output text
check "sagemaker endpoints" aws sagemaker list-endpoints --region "$REGION" \
  --query "Endpoints[].EndpointName" --output text
check "running ecs tasks" aws ecs list-clusters --region "$REGION" \
  --query "clusterArns[?contains(@,'${PREFIX}')]" --output text

echo
if [[ $FOUND -eq 0 ]]; then
  echo "PASS: no orphaned resources"
  exit 0
fi
echo "FAIL: $FOUND resource type(s) still present"
exit 1
