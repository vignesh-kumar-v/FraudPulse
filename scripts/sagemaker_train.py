#!/usr/bin/env python
"""Run the FraudPulse training set through a real SageMaker training job.

Phase 6's point is that the Terraform module provisions something that actually
works, not something that merely plans. This uploads the training data to the
bucket Terraform created, submits a job under the role Terraform created, waits
for it, and pulls the model artifact back.

Cost control, because this is a portfolio project and not a funded team:

  * AWS's built-in XGBoost image is used rather than a custom container - no
    ECR repo, no docker build, no push.
  * ml.m5.large, billed per second. A run over 413k x 31 takes a few minutes,
    which is roughly one cent.
  * ``MaxRuntimeInSeconds`` is set. A job that hangs stops billing at the cap
    instead of running until someone notices.
  * A *training job*, never an endpoint. An endpoint bills by the hour whether
    or not it serves a request, and forgetting one is the expensive mistake.

The local MLflow-registered model remains the one the API serves; this is the
"can it run on managed infrastructure" leg, and the two PR-AUCs are compared at
the end as a sanity check that the cloud path learned the same thing.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import boto3
import pandas as pd

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger
from fraudpulse.training.dataset import chronological_split, load_or_build

log = get_logger("sagemaker")

# AWS-managed XGBoost 1.7-1 image, us-east-1. Region-specific by design.
XGBOOST_IMAGES = {
    "us-east-1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1",
    "us-west-2": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1",
    "eu-west-1": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-xgboost:1.7-1",
}


def terraform_outputs(tf_dir: Path) -> dict:
    proc = subprocess.run(
        ["terraform", "output", "-json"], cwd=tf_dir, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not read terraform outputs from {tf_dir}. "
            f"Run `terraform apply` first.\n{proc.stderr}"
        )
    return {k: v["value"] for k, v in json.loads(proc.stdout).items()}


def _to_sagemaker_csv(X: pd.DataFrame, y: pd.Series) -> bytes:
    """SageMaker XGBoost wants headerless CSV with the label in column 0."""
    frame = pd.concat([y.rename("label").reset_index(drop=True),
                       X.reset_index(drop=True)], axis=1)
    buf = io.StringIO()
    frame.to_csv(buf, header=False, index=False)
    return buf.getvalue().encode()


def upload_splits(s3, bucket: str, prefix: str) -> tuple[str, str, dict]:
    split = chronological_split(load_or_build())
    train_key = f"{prefix}/train/train.csv"
    valid_key = f"{prefix}/validation/validation.csv"

    log.info("uploading train (%d rows) and validation (%d rows) to s3://%s/%s",
             len(split.X_train), len(split.X_valid), bucket, prefix)
    s3.put_object(Bucket=bucket, Key=train_key,
                  Body=_to_sagemaker_csv(split.X_train, split.y_train))
    s3.put_object(Bucket=bucket, Key=valid_key,
                  Body=_to_sagemaker_csv(split.X_valid, split.y_valid))
    return (f"s3://{bucket}/{prefix}/train/",
            f"s3://{bucket}/{prefix}/validation/",
            {"split": split.describe(), "n_features": split.X_train.shape[1]})


def submit(sm, *, job_name: str, image: str, role: str, instance_type: str,
           train_uri: str, valid_uri: str, output_uri: str, pos_weight: float,
           max_runtime: int) -> None:
    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
        RoleArn=role,
        HyperParameters={
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "num_round": "400",
            "max_depth": "6",
            "eta": "0.08",
            "subsample": "0.9",
            "colsample_bytree": "0.9",
            # Same imbalance handling as the local run, so the two PR-AUCs are
            # comparable rather than coincidentally similar.
            "scale_pos_weight": f"{pos_weight:.4f}",
            "early_stopping_rounds": "40",
        },
        InputDataConfig=[
            {
                "ChannelName": ch,
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated",
                }},
                "ContentType": "text/csv",
                "CompressionType": "None",
            }
            for ch, uri in (("train", train_uri), ("validation", valid_uri))
        ],
        OutputDataConfig={"S3OutputPath": output_uri},
        ResourceConfig={"InstanceType": instance_type, "InstanceCount": 1,
                        "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": max_runtime},
        Tags=[{"Key": "Project", "Value": "fraudpulse"},
              {"Key": "ManagedBy", "Value": "scripts/sagemaker_train.py"}],
    )


def wait(sm, job_name: str, poll_s: int = 20) -> dict:
    while True:
        d = sm.describe_training_job(TrainingJobName=job_name)
        status = d["TrainingJobStatus"]
        secondary = d.get("SecondaryStatus", "")
        if status in {"Completed", "Failed", "Stopped"}:
            return d
        log.info("  %s / %s ...", status, secondary)
        time.sleep(poll_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-dir", type=Path, default=settings.repo_root / "terraform")
    ap.add_argument("--max-runtime", type=int, default=1800,
                    help="Hard stop, in seconds. Caps the bill on a hung job.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Upload the data and print the job spec without submitting.")
    args = ap.parse_args()

    tf = terraform_outputs(args.tf_dir)
    region, bucket = tf["region"], tf["offline_store_bucket"]
    image = XGBOOST_IMAGES.get(region)
    if image is None:
        raise SystemExit(f"no built-in XGBoost image mapped for region {region}")

    s3 = boto3.client("s3", region_name=region)
    sm = boto3.client("sagemaker", region_name=region)

    train_uri, valid_uri, meta = upload_splits(s3, bucket, "training/input")
    output_uri = tf["training_output_uri"]

    split = chronological_split(load_or_build())
    pos = float(split.y_train.sum())
    pos_weight = (len(split.y_train) - pos) / max(pos, 1.0)

    job_name = f"fraudpulse-{int(time.time())}"
    log.info("job=%s image=%s instance=%s role=%s",
             job_name, image.rsplit("/", 1)[-1], tf["training_instance_type"],
             tf["sagemaker_role_arn"].rsplit("/", 1)[-1])

    if args.dry_run:
        log.info("dry run: not submitting. inputs=%s meta=%s", train_uri, meta)
        return 0

    t0 = time.perf_counter()
    submit(sm, job_name=job_name, image=image, role=tf["sagemaker_role_arn"],
           instance_type=tf["training_instance_type"], train_uri=train_uri,
           valid_uri=valid_uri, output_uri=output_uri, pos_weight=pos_weight,
           max_runtime=args.max_runtime)
    desc = wait(sm, job_name)
    wall = time.perf_counter() - t0

    status = desc["TrainingJobStatus"]
    billable = desc.get("BillableTimeInSeconds")
    metrics = {m["MetricName"]: m["Value"] for m in desc.get("FinalMetricDataList", [])}
    log.info("job %s in %.0fs wall, %ss billable", status, wall, billable)
    for k, v in metrics.items():
        log.info("  %s = %.5f", k, v)

    if status != "Completed":
        log.error("failure reason: %s", desc.get("FailureReason"))
        return 1

    artifact = desc["ModelArtifacts"]["S3ModelArtifacts"]
    local_pr_auc = _local_test_pr_auc()
    cloud_valid = metrics.get("validation:aucpr")

    # ml.m5.large on-demand, us-east-1
    est_cost = (billable or 0) / 3600 * 0.115
    result = {
        "job_name": job_name,
        "status": status,
        "region": region,
        "instance_type": tf["training_instance_type"],
        "wall_seconds": round(wall, 1),
        "billable_seconds": billable,
        "estimated_cost_usd": round(est_cost, 4),
        "model_artifact": artifact,
        "sagemaker_metrics": metrics,
        "local_test_pr_auc": local_pr_auc,
        "cloud_validation_aucpr": cloud_valid,
        **meta,
    }
    out = settings.reports_dir / "sagemaker_job.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info("artifact=%s  est cost=$%.4f  -> %s", artifact, est_cost, out)
    return 0


def _local_test_pr_auc() -> float | None:
    path = settings.reports_dir / "training_summary.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return max(r["test_pr_auc"] for r in d["results"])


def download_artifact(uri: str, dest: Path) -> Path:
    """Pull model.tar.gz back and unpack it, so 'it trained' is checkable."""
    bucket, key = uri.replace("s3://", "").split("/", 1)
    s3 = boto3.client("s3")
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "model.tar.gz"
    s3.download_file(bucket, key, str(tar_path))
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")
    log.info("unpacked %s -> %s", uri, dest)
    return dest


if __name__ == "__main__":
    sys.exit(main())
