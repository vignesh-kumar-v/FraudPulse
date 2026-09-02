# FraudPulse — Project Blueprint

Real-time fraud detection with a feature store, closing the train/serve-skew gap that a batch-only project (like TelcoFlow) can't demonstrate.

---

## 1. Elevator Pitch

Simulate a live transaction stream and flag fraud in near-real-time, using the **same feature definitions** at training time (offline, batch) and serving time (online, low-latency) — the specific, interview-relevant problem a feature store solves.

## 2. Why This Project (gaps it's designed to fill)

| Gap on current resume | How this project fills it |
|---|---|
| No streaming/event-driven system | Kafka as the ingestion layer |
| No feature store / train-serve-skew story | Feast, offline (batch) + online (Redis) |
| Only GCP represented in Cloud section | AWS (SageMaker/S3), provisioned via Terraform |
| Optuna only (sequential HPO) | Ray Tune (distributed HPO) — *stretch* |
| SHAP used only in an offline report | SHAP served live behind an API — *stretch* |

## 3. Scope: Core vs. Stretch

Per our scoping discussion — go deep on fewer pieces rather than shallow on all of them. Build in this order and **stop at Core if time is tight**; it's already a complete, defensible project on its own.

**Core (must finish, ~2–3 weeks):**
1. Kafka streaming ingestion (simulated)
2. Feature engineering + Feast (offline + online parity)
3. Model training (XGBoost/LightGBM) + real-time inference service
4. Evidently drift monitoring

**Stretch (only after Core is solid and *actually* verified working):**
5. Ray Tune distributed hyperparameter search
6. Terraform-provisioned AWS infra
7. SHAP explainability API

Do not start a stretch item until every Core item has a passing verification check (see Section 6). Partial stretch work with a broken Core story is worse than a finished Core with no stretch.

## 4. Architecture Overview

```
[Historical CSV data]                    [Simulated live transactions]
        |                                          |
        v                                          v
  Offline feature                            Kafka producer
  engineering (batch,                      (replays dataset as
  pandas/PySpark)                           a live event stream)
        |                                          |
        v                                          v
  Feast offline store                       Kafka consumer
  (Parquet/S3)                              --> compute online
        |                                     features on the fly
        v                                          |
  Model training                                   v
  (XGBoost/LightGBM)                        Feast online store
        |                                     (Redis)
        v                                          |
  Registered model                                 v
  (MLflow)                                  Inference service
        |                                   (FastAPI) <---- pulls same
        +---------- serves ---------------> feature definitions
                                                    |
                                                    v
                                          Evidently drift monitor
                                          (compares live feature
                                           distributions to training)
```

The point of this diagram: **one feature definition, two computation paths (batch for training, streaming for serving), reconciled by Feast.** That reconciliation — and being able to explain exactly where skew *could* creep in — is the actual skill being demonstrated, not any single box in the diagram.

## 5. Data

- **Dataset:** IEEE-CIS Fraud Detection (Kaggle) or PaySim (synthetic mobile money transactions). PaySim is smaller and easier to reason about for a first pass; IEEE-CIS is richer and more realistic if you want a harder feature-engineering story.
- **Simulating "live":** write a small producer script that reads the historical CSV in timestamp order and publishes one Kafka message per row with a short sleep between batches, so the consumer sees a realistic (sped-up) stream rather than an instant dump.

## 6. Build Phases — Steps and Verification

Each phase has a concrete "done" check. Don't move to the next phase until the current one's check passes.

### Phase 0 — Setup
- Repo scaffold, `docker-compose.yml` with Kafka (or Redpanda, a lighter Kafka-compatible alternative), Redis, and a Postgres/Parquet store for Feast's offline layer.
- **Verify:** `docker-compose up` brings up all services; a test producer/consumer pair can round-trip a message through Kafka.

### Phase 1 — Streaming ingestion
- Build the producer (replays dataset as a stream) and a consumer that writes raw events to a landing zone (Parquet on local disk or S3).
- **Verify:** run the full dataset through end-to-end; row counts match between source CSV and landed events (no silent drops) — this is exactly the kind of check that produced real bugs in your other projects, and it will here too.

### Phase 2 — Feature engineering + Feast
- Define entity: `account_id` (or `card_id`, depending on dataset).
- Features to implement (mix of point-in-time batch features and streaming aggregates):
  - Rolling transaction count, last 1h / 24h / 7d
  - Rolling transaction amount sum/mean, same windows
  - Time since last transaction for this account
  - Merchant-category frequency for this account
- Register these in a Feast feature repo (`feature_store.yaml`, `.py` feature definitions).
- Compute offline features in batch (pandas or PySpark) for training. Compute the *same* features online as events stream in, materialized into Redis via Feast.
- **Verify (the critical check):** pick 20 random (account, timestamp) pairs, compute each feature both via the offline path and the online path, and confirm they match. Log and explain any mismatch you find — that mismatch *is* the train/serve-skew story, and finding + explaining one honestly is more valuable to a resume bullet than everything matching perfectly on the first try.

### Phase 3 — Model training and inference service
- Train XGBoost and/or LightGBM on the offline feature set; tune with Optuna (you already know this from TelcoFlow — no need to relearn it here).
- Register the winning model in MLflow.
- Build a small FastAPI service: given an incoming transaction, pull the account's current features from Feast's online store (Redis), run inference, return a fraud score.
- **Verify:** p95 latency measured under a simple load test (e.g., `locust` or a basic loop with `httpx`), end-to-end from Kafka event to fraud score. Report the actual number — don't round up.

### Phase 4 — Drift monitoring
- Use Evidently to compare the live (streaming) feature distributions against the training-time distributions on a rolling window.
- Wire up a simple alert (log line or Slack webhook) when drift crosses a threshold on a key feature.
- **Verify:** intentionally shift the simulated stream's behavior (e.g., replay a different time slice, or inject synthetic distribution shift) and confirm the monitor actually fires.

### Phase 5 (stretch) — Ray Tune distributed HPO
- Swap the Optuna sequential search for Ray Tune running trials in parallel (can be local multi-process to start — doesn't require a real cluster to demonstrate the concept).
- **Verify:** compare wall-clock time for N trials, sequential vs. parallel, and report the actual speedup.

### Phase 6 (stretch) — Terraform-provisioned AWS
- Terraform module provisioning: S3 bucket (offline store), a SageMaker training job (or just an EC2/ECS task if SageMaker setup is too heavy for the timebox), and teardown via `terraform destroy`.
- **Verify:** `terraform apply` / `terraform destroy` cleanly create and tear down all resources with no orphaned infra — this is the actual thing that gets asked about in interviews ("what happens if apply fails halfway through?").

### Phase 7 (stretch) — SHAP explainability API
- Add an endpoint that returns the top-N SHAP feature contributions for a given fraud score, so an "analyst" could see *why* a transaction was flagged.
- **Verify:** contributions are computed fast enough to return synchronously (report the actual added latency vs. Phase 3's baseline).

## 7. Tech Stack Summary

| Layer | Tool | Status |
|---|---|---|
| Streaming ingestion | Kafka (or Redpanda) | Core |
| Feature store | Feast (offline: Parquet/S3, online: Redis) | Core |
| Batch feature computation | pandas / PySpark | Core |
| Model | XGBoost / LightGBM + Optuna | Core |
| Experiment tracking | MLflow | Core |
| Serving | FastAPI | Core |
| Drift monitoring | Evidently | Core |
| Distributed HPO | Ray Tune | Stretch |
| Infra as code | Terraform (AWS: S3, SageMaker/EC2) | Stretch |
| Explainability | SHAP (served live) | Stretch |

## 8. "Done" Definition

The project is resume-worthy at Core scope when all of the following are true, with real numbers behind each claim:

- [ ] End-to-end pipeline runs from simulated stream → feature computation → inference → response, with no manual steps
- [ ] Offline/online feature parity check has been run and any discrepancy found is understood and documented (not just "they matched")
- [ ] p95 inference latency is measured and reported
- [ ] Drift monitor has been proven to fire on an intentional distribution shift, not just left untested
- [ ] Model quality metric (PR-AUC, since fraud is class-imbalanced — accuracy is meaningless here) is reported against a clear baseline

## 9. Metrics to Track and Report Honestly

- **PR-AUC** (not plain AUC or accuracy — the dataset is highly imbalanced)
- p95 / p99 inference latency
- Feature parity mismatch rate (offline vs. online), if any
- Drift monitor true-positive rate on the injected shift test
- If you hit a bug or a result that's worse than expected — document it and the fix, the way you did with the FSDP bug in NanoLLM or the TensorRT silent failure in MLX-OCR. That kind of finding is what made those bullets strong; this project should aim for the same standard, not a clean success story.

## 10. Suggested Timeline (Core scope, ~2–3 weeks)

- **Days 1–2:** Phase 0 (setup) + Phase 1 (streaming ingestion)
- **Days 3–7:** Phase 2 (features + Feast) — this is the hardest and most important phase; don't rush it
- **Days 8–11:** Phase 3 (training + inference service)
- **Days 12–14:** Phase 4 (drift monitoring) + write-up / README / resume bullet draft
- **If ahead of schedule:** pick *one* stretch phase, not all three

## 11. Target Resume Bullet (draft — fill in real numbers once built)

> Simulated a live transaction stream via Kafka feeding a Feast feature store (online: Redis, offline: S3/Parquet), verifying offline/online feature parity to eliminate train-serve skew; served fraud predictions via FastAPI at [X]ms p95 latency; wired Evidently drift monitoring that [caught/flagged] injected distribution shift within [N] events.

Add Ray Tune / Terraform / SHAP clauses only if those phases were actually built and verified — don't pad the bullet with stretch items that stayed shallow.

## 12. Risks / Open Questions

- Feast's local/dev setup (Redis + a local offline store) is enough to prove the concept — a full production Feast deployment is out of scope and not needed for a resume project.
- If IEEE-CIS turns out too large/slow to iterate on locally, fall back to PaySim or a sampled subset — the feature-store story doesn't need the full dataset to be convincing.
- Decide early whether SageMaker (heavier setup, more resume-recognizable name) or plain EC2/ECS (lighter, faster to get working) is the better use of the stretch-phase time budget, given the timeline.
