# FraudPulse on AWS

Terraform module for the cloud training leg. Everything here was applied,
exercised with a real training job, and destroyed against account
`032614959255` in `us-east-1`; the numbers below are from that run, not from a
plan.

```bash
cd terraform
terraform init
terraform plan -out=tfplan
terraform apply tfplan

cd .. && python scripts/run_fargate_training.py     # build, push, train, verify

cd terraform && terraform destroy -auto-approve
../scripts/check_orphans.sh                          # the real teardown gate
```

## What it creates (19 resources)

| Resource | Why |
|---|---|
| S3 bucket + versioning + SSE + public-access-block | Feast offline store and training I/O |
| S3 lifecycle rule | Expires snapshots at 14 days, **and** non-current versions at 1 |
| IAM role for SageMaker + inline policy | Scoped to this one bucket, not `AmazonSageMakerFullAccess` |
| ECR repository + lifecycle policy | Trainer image; keeps the 3 most recent |
| ECS cluster, task definition, 2 IAM roles | Fargate training job |
| Security group (egress only) | The trainer serves nothing, so it has no ingress rule |
| 2 CloudWatch log groups, 7-day retention | SageMaker's own log group never expires by default |

Two IAM roles for ECS on purpose. The **execution** role is the ECS agent's —
pull the image, open the log stream — and runs before any project code does.
The **task** role is what the container itself gets, and it can reach exactly
one bucket. The container never holds ECR pull rights; the agent never holds
access to the data.

## SageMaker vs. Fargate

The blueprint allows either. This account got Fargate, for a reason worth
recording: **every one of the 142 SageMaker training-instance quotas is 0 on a
new AWS account.**

```
ResourceLimitExceeded: The account-level service limit
'ml.m5.large for training job usage' is 0 Instances
```

An increase for `L-611FA074` has been requested and is pending AWS review. The
SageMaker execution role and `scripts/sagemaker_train.py` are complete and will
work the moment it is granted. Fargate needs no increase (30 vCPU by default),
costs less, and runs the same job.

## The measured training run

| | |
|---|---|
| Task wall time | 22 s |
| Fit time inside the container | 23.8 s |
| Estimated cost | **$0.0003** |
| Cloud test PR-AUC | 0.16751 |
| Local test PR-AUC | 0.17164 |
| Delta | −0.00414 |

The remaining gap is hyperparameters, not skew: the local run is Optuna-tuned
over 40 trials, the container uses one fixed configuration. The comparison
exists because the *first* run came back at 0.14247 — see
[findings.md #11](../docs/findings.md), where the cloud path had silently zeroed
four categorical features.

## "What happens if apply fails halfway through?"

Demonstrated rather than described. An S3 bucket name that is globally taken
cannot be caught by any provider schema, so it fails at the API partway through
apply, after siblings have already been created:

```
aws_cloudwatch_log_group.created_before_the_failure: Creation complete after 1s
aws_s3_bucket.will_fail: Creating...
Error: creating S3 Bucket (s3): api error InvalidBucketName
```

Observed afterwards:

1. **State holds what succeeded.** `aws_cloudwatch_log_group.created_before_the_failure`
   is in `terraform state list`; the log group is really in AWS.
2. **The failed resource is absent from state.** No half-record — Terraform does
   not write a resource it could not create.
3. **Re-running apply is safe.** It skips the created resource and retries only
   the failed one. Terraform is converging, not replaying.
4. **Deleting the resource from the config makes it an orphan Terraform still
   owns.** `terraform plan` then proposes `0 to add, 0 to change, 1 to destroy`.

Point 4 is the one that actually bites: a resource created by a partial apply is
tracked, so removing the config **and** the state entry (or just deleting the
`.tf` file after a `terraform state rm`) is what leaves something billing that
nobody is watching.

An earlier attempt at this demo used `retention_in_days = 13`. The AWS provider
enumerates the valid set in its schema and rejected it during *plan* — the apply
never started. Worth knowing, but not the failure mode in question: schema
validation catches type errors, and only the API catches "this name is taken",
"this quota is 0", "this AZ has no capacity".

## Teardown

`Destroy complete!` is not the gate. It only means Terraform removed what it had
in state, and it says nothing about resources created outside Terraform — a log
group the service auto-created, an image CI pushed, a bucket a failed apply
never recorded. `scripts/check_orphans.sh` queries AWS directly across ten
resource classes:

```
$ terraform destroy -auto-approve
Destroy complete! Resources: 19 destroyed.        # 7.1s

$ ./scripts/check_orphans.sh
  clean   s3 buckets
  clean   iam roles
  clean   ecr repositories
  clean   ecs clusters
  clean   ecs task definitions
  clean   cloudwatch log groups
  clean   security groups
  clean   sagemaker training jobs (in progress)
  clean   sagemaker endpoints
  clean   running ecs tasks

PASS: no orphaned resources
```

The destroy ran against a bucket holding 6 objects and 181 MB, plus an ECR
repository with pushed images. Both refuse deletion when non-empty, which is the
most common cause of a half-finished destroy; `force_destroy` / `force_delete`
are set to `var.enable_force_destroy` for exactly that reason. It is a real
tradeoff and the variable documents it: destroy becomes reliable, and destroy
becomes able to delete data. Correct for a bucket rebuilt from local Parquet in
one command; wrong for anything you cannot regenerate.

## Deliberate omissions

**No SageMaker endpoint.** An endpoint bills by the hour whether or not it
serves a request. Training jobs and Fargate tasks bill for the seconds they run.

**Local state, not an S3 backend.** A remote backend is right for a team, but in
a single-module repo it is a bootstrapping paradox: the bucket holding the state
must exist before the config that creates buckets can run. The usual answer is a
separate bootstrap module applied once by hand — documented rather than
half-built.

**No NAT gateway.** ~$32/month to run, and the training task only needs egress.
`assignPublicIp: ENABLED` in the default VPC does the job for a batch task with
no ingress rule.

## Costs actually incurred

| | |
|---|---|
| S3 (181 MB, ~40 min) | < $0.001 |
| ECR (one ~400 MB image, ~1 hr) | < $0.001 |
| Fargate (2 runs, 22 s + 75 s, 1 vCPU / 4 GB ARM64) | $0.0013 |
| IAM, CloudWatch, security groups | $0 |
| **Total** | **under $0.01** |
