# Architecture

```
 IEEE-CIS train_transaction.csv (590,540 rows, 13,553 cards, 3.50% fraud)
                            |
                    fraudpulse prepare
                            |
                 data/processed/events.parquet
                            |
        +-------------------+--------------------+
        |                                        |
   OFFLINE PATH                            STREAMING PATH
   (training)                              (serving)
        |                                        |
        v                                        v
 features/offline.py                     streaming/producer.py
 vectorised, whole history               replays in timestamp order,
 searchsorted + prefix sums              keyed by card_id
 + sparse-table range max                        |
        |                                        v
        |                              Redpanda topic `transactions`
        |                                  6 partitions
        |                                  +--------+--------+
        |                                  |                 |
        |                                  v                 v
        |                     streaming/consumer_       streaming/consumer_
        |                     landing.py                features.py
        |                     raw -> Parquet            features/online.py
        |                     (replay + audit)          incremental keyed state
        |                                                     |
        v                                                     v
 feature_repo/data/                                   Feast PushSource
 card_features.parquet                                       |
        |                                                    v
        v                                              Redis (online store)
 Feast FileSource                                            |
 (duckdb offline store)                                      |
        |                                                    |
        +---------------> Feast registry <-------------------+
                     entity: card_id
                     FeatureView: card_stats (19 features)
                     OnDemandFeatureView: txn_ratios (4)
                     FeatureService: fraud_serving_v1
                            |
        +-------------------+-------------------+
        |                                       |
        v                                       v
 get_historical_features               get_online_features
 (point-in-time correct)               (sub-millisecond)
        |                                       |
        v                                       v
 training/train.py                      serving/app.py
 XGBoost + LightGBM                     FastAPI /score /explain
 Optuna (or Ray Tune)                   SHAP TreeExplainer
        |                                       ^
        v                                       |
 MLflow registry --------- model + ------------+
                     serving_metadata.json
                            |
                            v
                  monitoring/drift.py
                  Evidently: training distribution
                  vs. what the store actually served
```

## The one thing this diagram is about

Two computation paths, one feature definition, reconciled by Feast. The
interesting engineering is not any single box — it is that
`features/offline.py` and `features/online.py` are **independent
implementations of the same written spec** (`features/spec.py`), and
`features/parity.py` exists to catch them disagreeing.

They deliberately do not share code. If they did, the parity check would be a
tautology: it would always pass, and it would prove nothing about train/serve
skew. Every one of the seven bugs in [findings.md](findings.md) was found by a
check that could fail.

## Where skew can enter, and what stops it

| Skew source | What stops it |
|---|---|
| Different window arithmetic in batch vs. stream | `features/parity.py` — value-by-value comparison, exact for counters, `rtol=1e-9` for floats |
| Tied timestamps counted differently | `watermark` tie policy; the `arrival` policy's 0.0281% divergence is asserted by a test so the passing case can't go vacuous |
| Online store trailing the stream | Processing-time release timer + `finalize()`; regression test in `test_online_engine.py` |
| A stored feature that means something different when read later | `AccountState.snapshot()` takes no clock; read-time features live in the on-demand view |
| A write that reports success but stores nothing | `_assert_push_landed()` reads a value back after the first batch |
| Point-in-time join losing rows | Microsecond tiebreaker + a row-count assertion that refuses to train on a truncated set |
| Training and serving encoding categoricals differently | `serving_metadata.json` ships the category map and column order with the model |
| Batch and single-row on-demand code drifting apart | `test_ondemand.py` asserts they agree row-for-row |
| The model's input distribution moving after deployment | Evidently, referenced against the training distribution, verified against injected shift *and* a null control |

## Why these specific tools

**Redpanda over Kafka** — Kafka-API-compatible, one container instead of a
broker plus ZooKeeper/KRaft, ~1GB instead of ~4GB. Nothing in this project
touches an API that differs.

**DuckDB over the default dask offline store** — same Parquet files; the dask
point-in-time join took 70s on 5,000 entity rows and died with no traceback on
590,540. DuckDB does it in 16s. See [findings.md #6](findings.md).

**Redis for the online store** — what Feast's local provider supports, and the
latency budget for the store lookup is what the p95 number is mostly made of.

**6 partitions** — enough to make partition interleaving real (68% of messages
arrive out of global timestamp order), which is what surfaced
[findings.md #4](findings.md). Keying by `card_id` keeps per-card ordering
exact, which is the only ordering the feature engine actually needs.
