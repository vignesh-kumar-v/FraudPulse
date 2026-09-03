#!/usr/bin/env python
"""Phase 6 (stretch): run the training job on AWS Fargate, end to end.

Steps, all against what Terraform provisioned:
  1. upload the joined training Parquet to the offline-store bucket
  2. build the trainer image and push it to ECR
  3. run the ECS task and stream its CloudWatch logs
  4. pull the metrics back and compare cloud PR-AUC to the local number

Step 4 is the part that makes this more than a deployment exercise: if the two
PR-AUCs disagree, something about the cloud path differs from the local one and
the difference is worth finding.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import boto3

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger

log = get_logger("fargate")

TRAINING_KEY = "training/training_set.parquet"
OUTPUT_PREFIX = "training/output"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log.info("$ %s", " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    p = subprocess.run(cmd, text=True, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p


def terraform_outputs(tf_dir: Path) -> dict:
    p = subprocess.run(["terraform", "output", "-json"], cwd=tf_dir,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"terraform output failed in {tf_dir}:\n{p.stderr}")
    return {k: v["value"] for k, v in json.loads(p.stdout).items()}


def upload_training_set(s3, bucket: str) -> int:
    path = settings.processed_dir / "training_set.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `make train` first.")
    size = path.stat().st_size
    log.info("uploading %s (%.1f MB) -> s3://%s/%s", path.name, size / 1e6, bucket,
             TRAINING_KEY)
    s3.upload_file(str(path), bucket, TRAINING_KEY)
    return size


def build_and_push(repo_url: str, region: str, arch: str) -> str:
    registry = repo_url.split("/")[0]
    ecr = boto3.client("ecr", region_name=region)
    token = ecr.get_authorization_token()["authorizationData"][0]["authorizationToken"]
    user, password = base64.b64decode(token).decode().split(":", 1)

    _run(["docker", "login", "--username", user, "--password-stdin", registry],
         input=password)

    platform = "linux/arm64" if arch == "ARM64" else "linux/amd64"
    tag = f"{repo_url}:latest"
    # --provenance=false: buildx otherwise pushes an OCI image index with an
    # attestation manifest, and ECS refuses to pull it ("image manifest is not
    # supported"). The failure appears as a stopped task with no logs.
    _run(["docker", "buildx", "build", "--platform", platform,
          "--provenance=false", "--sbom=false",
          "-f", "docker/Dockerfile.trainer", "-t", tag, "--push", "."],
         cwd=settings.repo_root)
    return tag


def run_task(ecs, tf: dict) -> str:
    resp = ecs.run_task(
        cluster=tf["ecs_cluster"],
        taskDefinition=tf["ecs_task_definition"],
        launchType="FARGATE",
        count=1,
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": tf["ecs_subnets"][:2],
                "securityGroups": [tf["ecs_security_group"]],
                # Needed in a default VPC with no NAT gateway: without a public
                # IP the task cannot reach ECR and dies on image pull.
                "assignPublicIp": "ENABLED",
            }
        },
        tags=[{"key": "Project", "value": "fraudpulse"}],
    )
    if resp.get("failures"):
        raise RuntimeError(f"run_task failed: {resp['failures']}")
    arn = resp["tasks"][0]["taskArn"]
    log.info("task %s starting", arn.rsplit("/", 1)[-1])
    return arn


def wait_and_stream(ecs, logs, tf: dict, task_arn: str, poll: int = 10) -> dict:
    task_id = task_arn.rsplit("/", 1)[-1]
    stream = f"trainer/trainer/{task_id}"
    token, seen = None, 0

    while True:
        d = ecs.describe_tasks(cluster=tf["ecs_cluster"], tasks=[task_arn])["tasks"][0]
        status = d["lastStatus"]

        try:
            kw = {"logGroupName": tf["ecs_log_group"], "logStreamName": stream,
                  "startFromHead": True}
            if token:
                kw["nextToken"] = token
            ev = logs.get_log_events(**kw)
            for e in ev["events"]:
                print(f"    {e['message']}", flush=True)
                seen += 1
            token = ev["nextForwardToken"]
        except logs.exceptions.ResourceNotFoundException:
            pass  # stream does not exist until the container starts

        if status == "STOPPED":
            c = d["containers"][0]
            return {
                "task_id": task_id,
                "exit_code": c.get("exitCode"),
                "stopped_reason": d.get("stoppedReason"),
                "container_reason": c.get("reason"),
                "started_at": str(d.get("startedAt")),
                "stopped_at": str(d.get("stoppedAt")),
                "log_events": seen,
            }
        log.info("  task %s ...", status)
        time.sleep(poll)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-dir", type=Path, default=settings.repo_root / "terraform")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    tf = terraform_outputs(args.tf_dir)
    region, bucket = tf["region"], tf["offline_store_bucket"]
    s3 = boto3.client("s3", region_name=region)
    ecs = boto3.client("ecs", region_name=region)
    logs = boto3.client("logs", region_name=region)

    t_all = time.perf_counter()
    upload_bytes = 0 if args.skip_upload else upload_training_set(s3, bucket)
    if not args.skip_build:
        build_and_push(tf["ecr_repository_url"], region, "ARM64")

    t_task = time.perf_counter()
    outcome = wait_and_stream(ecs, logs, tf, run_task(ecs, tf))
    task_seconds = time.perf_counter() - t_task

    if outcome["exit_code"] != 0:
        log.error("task failed: %s", outcome)
        return 1

    metrics = json.loads(
        s3.get_object(Bucket=bucket, Key=f"{OUTPUT_PREFIX}/metrics.json")["Body"].read()
    )
    head = s3.head_object(Bucket=bucket, Key=f"{OUTPUT_PREFIX}/model.json")

    local = None
    lp = settings.reports_dir / "training_summary.json"
    if lp.exists():
        local = max(r["test_pr_auc"] for r in json.loads(lp.read_text())["results"])

    # 1 vCPU / 4 GB Fargate ARM64, us-east-1 on-demand
    cost = task_seconds / 3600 * (1 * 0.03238 + 4 * 0.00356)
    result = {
        **outcome,
        "task_wall_seconds": round(task_seconds, 1),
        "total_wall_seconds": round(time.perf_counter() - t_all, 1),
        "estimated_cost_usd": round(cost, 4),
        "training_parquet_bytes": upload_bytes,
        "model_artifact": f"s3://{bucket}/{OUTPUT_PREFIX}/model.json",
        "model_bytes": head["ContentLength"],
        "cloud_metrics": {k: v for k, v in metrics.items() if k != "feature_order"},
        "local_test_pr_auc": local,
        "cloud_vs_local_pr_auc_delta": (
            None if local is None else round(metrics["test_pr_auc"] - local, 5)
        ),
    }
    out = settings.reports_dir / "fargate_job.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    log.info("task %s in %.0fs, est $%.4f", outcome["exit_code"], task_seconds, cost)
    log.info("cloud test PR-AUC=%.5f  local=%s  delta=%s",
             metrics["test_pr_auc"], local, result["cloud_vs_local_pr_auc_delta"])
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
