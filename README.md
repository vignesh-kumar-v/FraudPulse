# FraudPulse

[![ci](https://github.com/vignesh-kumar-v/FraudPulse/actions/workflows/ci.yml/badge.svg)](https://github.com/vignesh-kumar-v/FraudPulse/actions/workflows/ci.yml)

Real-time fraud detection on a Feast feature store, built to answer one
question: **can you prove the features your model was trained on are the same
features it is served?**

Two independent implementations of one written feature spec — a vectorised
whole-history algorithm for training, an arrival-ordered keyed state machine for
serving — reconciled by Feast and checked against each other value-by-value on
every one of 590,540 real transactions.

They deliberately share no code. If they did, the parity check would be a
tautology. It has to be able to fail, and [it did, eleven
times](docs/findings.md).

---

## Results

Everything below is measured on the full IEEE-CIS dataset (590,540 transactions,
13,553 cards, 3.50% fraud, Dec 2017 – Jun 2018) through a real broker, a real
Redis, and — for the cloud leg — real AWS. Reproduce with `make verify`.

| Phase | Gate | Result |
|---|---|---|
| 1 | Landed rows == source rows | **590,540 == 590,540**, 0 malformed |
| 2 | Offline vs. online feature parity | **0 / 590,540** mismatched |
| 2 | Online store vs. offline, through Kafka | **0** mismatched, **20/20** Redis spot checks exact |
| 3 | Model beats a real baseline | **PR-AUC 0.1716** vs 0.0371 amount-only (**4.6×**) |
| 3 | p95 inference latency | **2.22 ms** end-to-end, 1.31 ms server-side |
| 4 | Drift fires on shift, not on noise | null **0.000**, temporal floor 0.391, **3/3** injections detected |

```
$ make verify
7/7 checks passed
```

### Train/serve skew, quantified

The streaming path and the batch path have to agree on tied timestamps, and
naively they do not:

| tie policy | rows mismatching | worst single feature error |
|---|---|---|
| `arrival` (naive) | 166 / 590,540 (0.0281%) | 7 transactions, $7,728 of windowed amount |
| `watermark` | **0** | — |

A test asserts the naive policy *still* diverges, so the passing case can never
quietly become vacuous.

### Latency

| load | rps | client p50 | client p95 | server p50 | **server p95** |
|---|---|---|---|---|---|
| 1 concurrent | 487 | 2.00 ms | 2.22 ms | 1.16 ms | **1.31 ms** |
| 8 concurrent | 1,093 | 6.45 ms | 12.52 ms | 2.03 ms | **3.87 ms** |
| 32 concurrent (4 client procs) | 2,141 | 14.15 ms | 24.15 ms | 3.97 ms | **8.74 ms** |
| 32 concurrent (1 client proc) | 511 | 34.72 ms | 202.94 ms | 1.44 ms | **2.17 ms** |

That last row is not the service failing. It is the load generator being
GIL-bound — the server's own self-timing stayed flat while the client's ballooned
9×. Splitting the *same* 32 in-flight requests across four processes took
throughput from 511 to 2,141 rps with nothing changed server-side.
[findings.md #8](docs/findings.md).

SHAP adds **+4.37 ms** to p95 (2.22 → 6.59 ms at concurrency 1).

### Model

| | test PR-AUC | ROC-AUC |
|---|---|---|
| Prevalence (random) | 0.0348 | 0.500 |
| Amount only | 0.0371 | — |
| LightGBM (40 Optuna trials) | 0.1602 | 0.790 |
| **XGBoost (40 Optuna trials)** | **0.1716** | 0.802 |

PR-AUC, not accuracy — at 3.5% fraud, "always legit" scores 96.5% accurate and
catches nothing. The split is chronological; validation PR-AUC is 0.212 and test
is 0.172, and that gap is real six-month drift in the data, not noise.

Only 26 velocity/aggregate features are used, no `V*`/`C*`/`D*` columns. The
point here is the feature store, not a Kaggle score.

### Stretch phases

**Ray Tune** — 20 trials, identical space and sampler, threads per trial capped
so the comparison measures parallelism rather than oversubscription:
149.0s → 90.2s, **1.65×**. Not 6× despite 6-way concurrency, because XGBoost
`hist` already saturated all 15 cores sequentially.
[findings.md #10](docs/findings.md).

**Terraform on real AWS** — 19 resources applied, a real Fargate training job
run against them (22s, **$0.0003**, cloud PR-AUC 0.16751 vs local 0.17164), then
destroyed in 7.1s with a 181 MB non-empty bucket, verified by a direct AWS query
across ten resource classes: **no orphaned resources**. Total spend under $0.01.
Includes a *demonstrated* answer to "what if apply fails halfway".
[terraform/README.md](terraform/README.md).

**SHAP** — served synchronously behind `/explain`, latency cost measured above.

---

## What actually went wrong

Eleven bugs, all written up with the measurement that caught each one in
**[docs/findings.md](docs/findings.md)**. The ones worth knowing about:

**`feast.push(to="online")` silently wrote nothing.** `PushMode` is a plain
`Enum`, so the string matched neither branch. No exception, no warning, the
counters kept climbing, Redis stayed empty.

**Partition interleaving + timestamp-based write dedup = silent data loss.**
Six partitions mean 68.3% of messages arrive out of global timestamp order
(measured; worst backward jump 8,288 days). Stamping online writes with the
global clock made Feast's staleness guard drop every later write for any card a
fast partition had touched. Both mechanisms are individually correct.

**Feast's point-in-time join deleted 166 training rows** — one from every pair of
same-card same-second transactions — and those rows were **8.4% fraud against a
3.50% base rate**, so it was preferentially deleting the positive class.

**The watermark that fixed tie skew stranded every card's newest event.** For a
velocity-based fraud feature that is exactly the wrong transaction to lose.

**A stored feature that was correct when written and wrong at every read after.**
`seconds_since_last_txn` is an age. Ages do not survive being stored.
`AccountState.snapshot()` now takes no clock at all, so the class of bug is
structurally excluded.

**The cloud training path zeroed four features, including the most important
one.** `pd.to_numeric(errors="coerce").fillna(0.0)` across every column turns
strings into a constant. Exit code 0, right row count, no NaNs, −0.029 PR-AUC.
Caught only because the deploy script compares cloud metrics to local ones.

**The drift monitor's "no shift" control was just a later time window**, so it
fired on the dataset's own seasonality and looked like a 30% false-positive rate.

---

## Architecture

```
 IEEE-CIS CSV ──► events.parquet ──┬──► features/offline.py ──► Feast FileSource
                                   │    vectorised, whole history      (DuckDB)
                                   │    searchsorted + prefix sums          │
                                   │    + sparse-table range max            │
                                   │                                        ▼
                                   │                          get_historical_features
                                   │                            (point-in-time)
                                   │                                        │
                                   └──► Kafka (Redpanda, 6 parts) ──┐       ▼
                                        keyed by card_id            │  XGBoost/LightGBM
                                                                    │  Optuna │ Ray Tune
                        ┌───────────────────────────────────────────┤       │
                        ▼                                           ▼       ▼
              consumer_landing.py                        consumer_features.py   MLflow
              raw ──► Parquet                            features/online.py     registry
              (replay + audit)                           incremental state         │
                                                                 │                 │
                                                                 ▼                 │
                                                          Feast PushSource         │
                                                                 │                 │
                                                                 ▼                 ▼
                                                          Redis ──► FastAPI /score /explain
                                                                            │
                                                                            ▼
                                                              Evidently drift monitor
```

Detail and the full skew-source table in
**[docs/architecture.md](docs/architecture.md)**.

The single most important thing in this repo is
[`src/fraudpulse/features/`](src/fraudpulse/features/): `spec.py` is the written
contract, `offline.py` and `online.py` are the two independent implementations,
and `parity.py` is what stops them drifting apart.

---

## Quick start

Needs Docker and Python 3.11.

```bash
make setup          # .venv + dependencies
make up             # redpanda, redis, mlflow, console
make smoke          # Phase 0: kafka round-trip from the host

make data           # kaggle download (needs competition rules accepted)
make prepare        # -> data/processed/events.parquet

make topic produce  # Phase 1: replay 590k transactions onto kafka
make land           # Phase 1: land raw events, assert the row count

make build-offline  # Phase 2: batch features + feast apply
make features       # Phase 2: stream -> incremental features -> redis
make parity         # Phase 2 verify: in-process AND end-to-end

make train          # Phase 3: feast historical join, optuna, mlflow registry
make serve          # Phase 3: FastAPI on :8000
make loadtest       # Phase 3 verify: p50/p95/p99
make drift          # Phase 4 verify: injected shift vs. null control

make verify         # every gate, one table
```

No dataset and no Docker? `make test` still runs — 47 tests build their fixtures
from a synthetic generator that deliberately includes duplicate timestamps, long
idle gaps and single-transaction cards.

`make e2e` runs the whole thing from nothing.

### Try the API

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "transaction_id": 1, "card_id": "card_7919", "amount": 950.0,
  "product_cd": "C", "card_network": "visa", "card_type": "credit",
  "email_domain": "gmail.com", "explain": true
}' | jq
```

```json
{
  "fraud_probability": 0.709,
  "is_fraud_pred": true,
  "model_version": "xgboost:v3",
  "feature_source": "online_store",
  "latency_ms": 2.1,
  "explanation": [{"feature": "product_cd", "value": 1.0, "contribution": 1.84}, ...]
}
```

`GET /features/{card_id}` shows what the online store currently holds for a card.
`GET /metrics` exposes per-stage Prometheus histograms — one end-to-end number
tells you the service is slow but not whether Redis, the on-demand maths or the
model is responsible.

---

## Stack

| Layer | Choice | Why this one |
|---|---|---|
| Streaming | Redpanda | Kafka API, one container instead of broker + KRaft |
| Feature store | Feast — Redis online, Parquet/DuckDB offline | The offline/online parity story is the project |
| Offline engine | DuckDB | The default dask store took 70s on 5k rows and died on 590k |
| Model | XGBoost / LightGBM + Optuna | PR-AUC on a 3.5%-positive problem |
| Tracking | MLflow | Registry ships `serving_metadata.json` with every model |
| Serving | FastAPI + Uvicorn | Sync handlers; the bottleneck was never here |
| Monitoring | Evidently | Method-aware drift thresholds, watchlist + share rule |
| Distributed HPO | Ray Tune | Measured against sequential, honestly |
| IaC | Terraform → S3, ECR, ECS Fargate, IAM, CloudWatch | Applied, trained on, destroyed clean |
| Explainability | SHAP TreeExplainer | Served synchronously, cost measured |

---

## Layout

```
src/fraudpulse/
  features/     spec.py, offline.py, online.py, ondemand.py, parity.py, timeline.py
  streaming/    producer.py, consumer_landing.py, consumer_features.py, topics.py
  training/     dataset.py, train.py, tune_ray.py
  serving/      app.py, model_store.py, loadtest.py
  monitoring/   drift.py, alert.py
  cli.py, config.py, schema.py, verify.py, feast_repo.py, status.py
feature_repo/   feature_store.yaml, definitions.py
docker/         Dockerfile.trainer, train_entrypoint.py
terraform/      main.tf, ecs.tf, variables.tf, outputs.tf, README.md
scripts/        smoke_kafka.py, verify_parity.py, sagemaker_train.py,
                run_fargate_training.py, check_orphans.sh
docs/           architecture.md, findings.md, blueprint.md
tests/          47 tests, no docker or dataset required
reports/        every number in this README, as JSON
```

---

## Honest limitations

- **The online feature engine keeps state in-process**, like Flink keyed state.
  Durability comes from replaying the Parquet landing zone, not from a
  checkpoint. Fine for one consumer; a real deployment needs RocksDB-backed
  state or Flink itself.
- **`release_after_s` trades staleness against tie-correctness.** A tie whose
  halves are more than 5 wall-clock seconds apart will be miscounted. There is
  no setting that gives both.
- **PR-AUC 0.172 is not a competitive IEEE-CIS score.** Deliberately: only
  card-velocity aggregates are used, no `V*`/`C*`/`D*` columns. The claim being
  made is about feature parity, not leaderboard rank.
- **Terraform state is local.** A remote backend is a bootstrapping paradox in a
  single-module repo; explained rather than half-built.
- **SageMaker is written but unrun** — the account's training-instance quotas are
  all 0 and the increase is pending. Fargate ran the job instead.
- **No authentication on the API.** Out of scope, and saying so is better than a
  token check that looks like security.
