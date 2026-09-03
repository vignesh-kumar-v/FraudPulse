# Findings — bugs found while building, and what they cost

Every entry here was found by a check that was written to be *able* to fail, not
by reading the code and thinking hard. They are listed in the order they were
found. Numbers are measured on the local stack (Redpanda 6 partitions, Redis 7,
40k-event synthetic replay unless stated otherwise).

---

## #1 — `feast.FeatureStore.push(to="online")` silently writes nothing

**Symptom.** The feature consumer logged `pushed=3901` and exited 0. Redis
contained 0 keys.

**Cause.** `FeatureStore.push` takes `to: PushMode`. `PushMode` is a plain
`enum.Enum`, not a `str`-mixin enum:

```python
>>> from feast.data_source import PushMode
>>> "online" == PushMode.ONLINE
False
```

The method body is two `if` statements comparing `to` against `PushMode.ONLINE`
and `PushMode.ONLINE_AND_OFFLINE`. A string matches neither. Both branches are
skipped, `pushed_fv_names` stays empty, and the function returns `None` — no
exception, no warning, no return value to check.

**Why it survived.** Everything upstream succeeded. The push loop ran, the
counters incremented, the log line was truthful about how many rows it had
*handed to* Feast. Nothing in the pipeline ever read a value back.

**Fix.** Pass the enum, and — more importantly — stop trusting the call:
`_assert_push_landed()` reads one entity back out of the online store after the
first batch and raises if it is `None`. One extra Redis round-trip per run.

> A write path with no read-back is not verified, it is merely uncontradicted.

---

## #2 — The watermark that fixed tie skew stranded every card's newest event

**Symptom.** After fixing #1, the blueprint's 20-pair Redis spot check failed
20/20. Every card's stored `txn_count_lifetime` was exactly one below the truth,
and its stored aggregates were missing exactly one transaction.

**Cause.** The `watermark` tie policy buffers events at a card's current maximum
timestamp and only folds them into state when a strictly later event arrives —
that is what makes the streaming path agree with the batch path's
`searchsorted(..., side="left")` on tied timestamps. But a card's *last* event
has no successor. It sits in the buffer forever and never reaches Redis.

**Why it matters more than it looks.** The stranded event is always the newest
one. For a velocity-based fraud feature that is precisely the transaction you
cannot afford to be missing — the online store was systematically blind to the
most recent activity on every card in the system.

**Fix.** A processing-time release timer (`OnlineFeatureEngine.release_idle`):
if nothing newer has arrived for a card in `release_after_s` wall-clock seconds,
commit the buffered event anyway. `finalize()` releases everything at end of
stream.

**Tradeoff, stated honestly.** `release_after_s` (default 5s) bounds online-store
staleness, and a tied pair whose halves are separated by more than that window
will be miscounted. There is no setting that gives both exact tie semantics and
zero staleness; the knob is the design.

---

## #3 — `seconds_since_last_txn` was correct when written and wrong ever after

**Symptom.** Redis returned `seconds_since_last_txn = 299491373` (≈9.5 years)
for cards whose last transaction was recent.

**Cause.** The feature was computed as `now - last_txn_ts` at *write* time and
stored as a number. An age is only true at the instant it is computed. Every
read after that returns a value that is stale by exactly how long the row has
been sitting there — and there is nothing in the value itself to reveal it.

**Fix.** The online store now holds `last_txn_unixtime`, an absolute timestamp
that means the same thing forever. `seconds_since_last_txn` moved to the
on-demand feature view and is subtracted against the *request's* clock. Feast's
`RequestSource` gained an `event_unixtime` field to carry it.

**Generalisation, now enforced by a test.** `AccountState.snapshot()` takes no
timestamp argument at all. Anything that depends on *when you ask* cannot be
materialised, by construction rather than by discipline.

---

## #4 — Stamping online writes with a global clock made Feast drop them silently

**Symptom.** After #2 and #3 were fixed, the Redis spot check still failed 9/20.
Some cards were badly stale — `card_00016` held `txn_count_lifetime = 20` when
the truth was 272, and its stored state was frozen at an event twenty years
earlier in simulated time. The staleness was not uniform: 11 cards were perfect,
9 were wildly wrong.

**Cause — two facts that are individually fine and jointly fatal.**

*Fact one:* the topic has 6 partitions. The producer emits in global timestamp
order, but a consumer reads partitions in whatever order data arrives. Measured
on the real topic:

```
messages inspected      : 20000
timestamp regressions   : 13666 (68.3%)
worst backward jump     : 8288 days
```

68% of messages arrive with an event time *earlier* than the running maximum.
Per-card ordering is still perfect (`out_of_order=0`) because the producer keys
by `card_id` — but the global stream clock is meaningless.

*Fact two:* Feast's Redis online store skips any write whose `event_timestamp`
is `<=` the one already stored for that entity. This is a correct and desirable
staleness guard.

The bug was stamping each online write with the *global* maximum event time.
Once a slow-partition card was stamped with a fast partition's clock, every
subsequent write for that card looked stale and was silently discarded. The
consumer's `pushed=` counter kept climbing; Redis stopped changing.

**Fix.** Stamp each row with **that card's own** newest committed event time
(`AccountState.committed_ts`), which is monotonic per card by construction. Also
sort by event time before de-duplicating a push batch — "last in the list" is not
"latest in time" when partitions interleave.

**Aftermath.** 20/20 spot checks exact; 40,000/40,000 events matching end-to-end
through the real broker.

**The transferable lesson.** Two mechanisms that are each correct in isolation —
partition-level parallelism and timestamp-based write deduplication — combined
into silent data loss, and the only visible signal was a counter that had never
been wrong before. This is what "the global event clock is not a thing in a
partitioned stream" means in practice.

---

## Non-bug: the 0.30% tie mismatch is real and is kept

`tests/test_feature_parity.py::test_arrival_policy_diverges_on_duplicate_timestamps`
asserts that the *naive* streaming policy still disagrees with the batch path:

```
tie_policy=arrival    rows with >=1 mismatched feature: 12 (0.3000%)
tie_policy=watermark  rows with >=1 mismatched feature: 0  (0.0000%)
```

That test exists so the passing case cannot quietly become vacuous. If the
fixture ever stops producing duplicate timestamps, the "0 mismatches" result
would still be reported — and would no longer mean anything.

---

## #5 — Feast's point-in-time join silently deleted 166 training rows, 2.4x enriched in fraud

**Symptom.** `get_historical_features` on 5,000 entity rows returned 4,995. No
warning.

**Cause.** The offline join ends with

```python
df.drop_duplicates(all_join_keys + [entity_df_event_timestamp_col], keep="last")
```

Two transactions on the same card in the same second are indistinguishable to
that predicate, so one of them disappears. `transaction_id` is unique and does
not help — it is not a join key.

**Scale, on the full dataset.**

```
total rows        : 590540
tied (card,ts) grp: 146
rows Feast drops  : 166  (0.0281%)
fraud rate in the dropped rows : 8.43%   (base rate 3.50%)
```

0.028% is small. That it is **2.4x enriched in the positive class** is not: the
join was preferentially deleting the examples the model most needs, and would
have kept doing so silently forever.

**Fix.** `features/timeline.py` adds a microsecond-scale tiebreaker — the row's
rank among its card's transactions in that second, ordered by `transaction_id` —
applied by one shared function to *both* sides of the join, so a feature row and
its own entity row still line up exactly. Plus a row-count assertion in
`build_training_set` that refuses to train on a truncated set.

Microseconds rather than nanoseconds: Feast normalises to Arrow
`timestamp[us]`, and a nanosecond offset raises
`ArrowInvalid: Casting from timestamp[ns] to timestamp[us] would lose data`.

---

## #6 — The default offline store took 70s on 5k rows and died on 590k with exit code 0

**Symptom.** `make train` printed one log line and exited 0. No traceback, no
error, no model. The dask-backed point-in-time join had taken the process down.

**Measurements.** 5,000 entity rows: 70.3s. 590,540 entity rows: process
disappears.

**Fix.** Switched `offline_store.type` from the default to `duckdb` (Feast's
ibis/DuckDB backend). Same Parquet files, same feature definitions, only the
join engine changes:

| entity rows | dask | duckdb |
|---|---|---|
| 5,000 | 70.3s | <1s |
| 100,000 | — | 2.1s |
| 590,540 | dies, exit 0 | 16.3s |

One wrinkle worth knowing: the dask store resolves a relative `FileSource` path
against the feature repo, and the duckdb store resolves it against the caller's
cwd. A relative path therefore works under `feast apply` and fails from
anything else. `definitions.py` now builds an absolute path from `__file__`.

---

## #7 — MLflow client 3.15 against server 2.16: a bare 404, after everything else succeeded

**Symptom.** Every parameter and metric logged fine. Model registration then
failed with

```
API request to endpoint /api/2.0/mlflow/logged-models failed with error code 404
```

**Cause.** The MLflow 3 client uses the logged-models API, which a 2.x server
does not serve. The failure lands at the very end, after a full tuning run.

**Fix.** Pinned the server image to the client's major (`v3.15.2`).

A second one behind it: with `--default-artifact-root /mlflow/artifacts`, the
server hands the client its own *container* path and the client — running on the
host — tries to `os.makedirs()` it, giving
`OSError: [Errno 30] Read-only file system: '/mlflow'`. Fixed with
`--serve-artifacts --artifacts-destination`, which makes the server proxy
artifact I/O and hand out `mlflow-artifacts:/` URIs instead. Note that an
experiment created before the switch keeps its old artifact location, so the
existing volume has to be dropped for the change to take.

---

## #8 — The load test was measuring the load generator

**Symptom.** Client-observed p95 climbed 2ms → 12ms → 88ms → 280ms as
concurrency went 1 → 8 → 16 → 32. The obvious reading: the service falls over
under load.

**The tell.** The server's own self-timing, returned on every response, stayed
flat the whole way:

| concurrency | client p95 | **server p95** | rps |
|---|---|---|---|
| 1 | 2.23 ms | 1.33 ms | 484 |
| 8 | 12.52 ms | 3.87 ms | 1,093 |
| 32 (1 process) | 202.94 ms | **2.17 ms** | 511 |

A service that was actually saturating would show its own work getting slower.
This one was doing 2.2ms of work per request while callers waited 200ms.

**Confirmation.** Same server, same 32 requests in flight, load split across 4
client processes instead of 1:

| 32 in flight | rps | client p95 |
|---|---|---|
| 1 process × 32 | 511 | 202.94 ms |
| 4 processes × 8 | **2,054** | **23.39 ms** |

4x the throughput and 9x better tail latency, with nothing changed on the server.
One Python process driving 32 concurrent HTTP requests is GIL-bound on JSON
serialisation and asyncio bookkeeping.

**Two real fixes came out of it anyway.** Caching the Feast feature service at
startup instead of resolving it per request took server p50 from 28.31ms to
1.65ms — that one was genuine. And `features_used` (31 floats) is now opt-in via
`include_features`, which moved single-process c=8 throughput from 850 to 1,201
rps.

**Lesson.** Always have the server time itself. Without that number, the
client-side curve is unfalsifiable and points at the wrong component.

---

## #9 — The drift monitor's "no shift" control was not a control

**Symptom.** The Phase 4 experiment failed: the unshifted window alerted, 7 of
23 columns drifting at share 0.304. Read at face value, a 30% false-positive
rate.

**Cause.** It was not a false positive. The "unshifted" window was simply *later
in time* — IEEE-CIS spans December 2017 to June 2018 and genuinely drifts across
those six months. The monitor was right. The experiment was wrong: it had no way
to distinguish "the monitor over-fires" from "the data moved".

A second, quieter problem in the same design: the reference was the whole
training set, which *contains* the window being tested against it.

**Fix.** Three comparisons instead of one:

| scenario | reference | current | drift share | verdict |
|---|---|---|---|---|
| `null` | random half of the train slice | the other half | **0.000** | quiet — correct |
| `temporal` | train slice | later slice, untouched | 0.391 | alerts — real seasonality, reported as the floor |
| `amount` | train slice | later slice, amounts ×4 | 0.783 | clears the floor |
| `velocity` | train slice | later slice, gaps ÷20 | 0.652 | clears the floor |
| `product` | train slice | later slice, mix collapsed | 0.478 | clears the floor |

The null control is now two random halves of the *same period*, so it isolates
the monitor's own behaviour: 0 of 23 columns, share 0.000. Passing requires the
null quiet, every injection alerting, **and** every injection exceeding the
temporal floor — otherwise "it fired" would only mean "time passed".

**Worth stating plainly:** this project's own model degrades across that same
drift. Validation PR-AUC 0.212, test PR-AUC 0.172 — measured on a chronological
split, so the gap is that six-month movement showing up in the metric.

---

## #10 — Distributed HPO bought 1.65x, not 6x, and the reason matters

**Setup.** 20 trials, identical search space and identical Optuna TPE sampler,
same 15-core machine. Only the execution changes: sequential Optuna, versus Ray
Tune with `max_concurrent_trials=6` and each trial capped at
`n_jobs = cores // concurrency = 2`.

**Result.**

```
sequential (optuna)  149.0s   best valid PR-AUC 0.20885
parallel   (ray)      90.2s   best valid PR-AUC 0.21035
                    -> 1.65x wall-clock
```

**Why it is not 6x, and why that was predictable.** XGBoost's `hist` tree method
already parallelises across cores. The sequential run was *not* leaving 14 cores
idle — it was using all 15 for every trial. Running six trials at once does not
add compute, it partitions the same compute six ways. The gain comes only from
the parts of a trial that do not parallelise well (data setup, early-stopping
bookkeeping, the tail of each boosting round), plus better core utilisation
during those gaps.

**What would have made the number look better, dishonestly.** Leaving `n_jobs`
at its default so each of the six trials grabbed all 15 cores. That produces
6x oversubscription: every trial runs slower, but the *sequential* baseline
would also have to be re-run under the same handicap to be comparable, and it
would not be. Capping threads per trial is what makes 1.65x a real number.

**Where Ray Tune would actually pay.** Trials that are individually
single-threaded, or a real multi-node cluster where the extra cores are
genuinely extra. On one box running an already-parallel learner, the honest
answer is that it is worth 1.65x and about 10 seconds of startup.

---

## #11 — The cloud training path silently zeroed four features, including the most important one

**Symptom.** The Fargate training job completed, exit code 0, model artifact in
S3, no warning anywhere. The only signal was one line the runner prints:

```
cloud test PR-AUC=0.14247  local=0.17164  delta=-0.02918
```

**Cause.** The container's feature preparation was one convenient line:

```python
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
```

Four of the 31 columns — `product_cd`, `card_network`, `card_type`,
`email_domain` — are strings. `errors="coerce"` turns each into `NaN`, and
`fillna(0.0)` turns every `NaN` into the same `0.0`. The result is four constant
columns. The model trained on 27 real features and four dead ones, and XGBoost
handled it exactly as it should: it ignored them and reported a perfectly valid
score for a worse model.

`product_cd` was the single highest-importance feature in the local model
(gain 0.273).

**Why nothing caught it.** Every guard in the pipeline was satisfied. The row
count was right. No NaNs reached the model. No exception, no warning, exit 0.
`errors="coerce"` is a request to convert failures into silence, applied to
columns that were always going to fail.

**Fix, in two parts.**

1. The container ordinal-encodes the categoricals from the train slice, the same
   way `training/dataset.py` does. Cloud PR-AUC 0.14247 → **0.16751**, a delta
   of −0.00414 against local, which is now fully explained by the local run
   being Optuna-tuned over 40 trials while the container uses one fixed config.
2. `_assert_no_dead_features()` aborts if any feature is constant across the
   training slice. A constant feature is almost never intentional — it is what a
   botched encoding, a failed join, or a missing column looks like from the
   inside. That check would have caught this on the first run rather than in the
   PR-AUC comparison afterwards.

**The reason it was findable at all** is that `run_fargate_training.py` compares
the cloud metric to the local one on every run. A deployment script that only
reported "task succeeded" would have shipped this.
