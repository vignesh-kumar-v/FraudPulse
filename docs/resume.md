# Resume bullets — with the run behind each number

Every figure here is in `reports/*.json` and reproducible with `make verify`.
Nothing is rounded up.

---

## The one-line version

> Built a real-time fraud detection pipeline (Kafka → Feast → FastAPI) that
> verifies offline/online feature parity across all 590,540 transactions,
> serving predictions at 2.27 ms p95; found and fixed 11 silent-failure bugs,
> including a partition-ordering interaction that made the feature store
> silently drop writes.

## The three-line version

> **FraudPulse** — Streamed 590,540 IEEE-CIS transactions through Redpanda into
> a Feast feature store (Redis online, DuckDB/Parquet offline), implementing the
> same 19-feature spec twice — vectorised for training, incremental for serving —
> and proving **0 mismatches across all 590,540 events** end-to-end through the
> broker, versus 166 under a naive streaming implementation.
>
> Served XGBoost predictions via FastAPI at **2.27 ms p95** (1.35 ms server-side,
> 2,054 rps at 32 concurrent), **PR-AUC 0.172 vs 0.037** for the strongest
> trivial baseline on a 3.5%-positive problem; added SHAP explanations for
> **+4.5 ms**.
>
> Wired Evidently drift monitoring verified against three injected shifts **and**
> a null control (0.000 drift share on same-period data), and provisioned the
> cloud training path with Terraform on AWS — applied, ran a Fargate job, and
> destroyed with **zero orphaned resources**, all for under $0.01.

---

## Bullets by theme

Pick the ones that match the role.

**Streaming / data engineering**
> Replayed 590,540 transactions through a 6-partition Redpanda topic keyed by
> card, landing them to Parquet with an exact row-count gate (590,540 in,
> 590,540 out, 0 malformed) and a per-card ordering assertion that stayed at 0
> violations while 68.3% of messages arrived out of *global* timestamp order.

**Feature store / train-serve skew** — the headline
> Implemented one written feature spec twice on purpose — a whole-history
> vectorised algorithm (binary search + prefix sums + sparse-table range max)
> for training, an arrival-ordered keyed state machine for serving — sharing no
> code, so the parity check could actually fail. It did: a naive streaming
> implementation diverged on 166 of 590,540 rows (0.0281%), off by up to 7
> transactions and $7,728 of windowed amount on a single feature vector. A
> watermark tie policy took it to 0, verified end-to-end through the real broker
> and Redis.

**Debugging / silent failures** — the strongest one
> Diagnosed a data-loss bug arising from two individually-correct mechanisms:
> Kafka partition parallelism (68.3% of messages out of global timestamp order,
> worst backward jump 8,288 days) combined with Feast's timestamp-based write
> deduplication, which silently discarded every subsequent online-store write
> for any card that a fast partition's clock had touched. Counters kept
> incrementing; Redis stopped changing. Fixed by stamping writes with each
> card's own committed event time.

**Data quality**
> Found that Feast's point-in-time join was silently deleting 166 training rows
> — one from every pair of same-card same-second transactions — and that those
> rows were **8.4% fraud against a 3.50% base rate**, i.e. 2.4× enriched in the
> positive class. Fixed with a shared microsecond tiebreaker applied to both
> sides of the join, plus a row-count assertion that refuses to train on a
> truncated dataset.

**ML engineering**
> Trained XGBoost and LightGBM through Feast's `get_historical_features` (not a
> pandas merge, so a renamed feature breaks training instead of silently serving
> a stale model), tuned with 40 Optuna trials against a chronological split, and
> registered the winner in MLflow with the category map and column order shipped
> alongside it. Test PR-AUC **0.1716** vs 0.0371 amount-only and 0.0348
> prevalence — 4.6× the strongest trivial baseline on a 3.5%-positive problem.

**Serving / performance**
> Served predictions from Redis-backed features at **2.27 ms p95** end-to-end
> (1.35 ms server-side), scaling to 2,054 rps. Caught that an apparent
> saturation at 32 concurrent (p95 203 ms) was the *load generator* being
> GIL-bound, not the service — the server's own per-stage timing stayed flat at
> 2.17 ms — and confirmed it by splitting the same in-flight load across four
> client processes for a 4× throughput gain with no server change.

**Monitoring**
> Verified Evidently drift detection adversarially: a null control (two random
> halves of the same period) must stay silent at 0.000 drift share, a temporal
> control quantifies the dataset's own six-month drift at 0.391 as a floor, and
> all three injected shifts — amounts ×4, inter-arrival gaps ÷20, product mix
> collapsed — must exceed it. The first version of this experiment used a later
> time window as its "no shift" control and read the data's real seasonality as
> a 30% false-positive rate.

**Distributed HPO** *(only if asked about scale)*
> Measured Ray Tune against sequential Optuna over an identical search space
> with per-trial threads capped to prevent oversubscription: 149.0 s → 90.2 s,
> **1.65×** on 6-way concurrency — not 6×, because XGBoost `hist` already
> saturates all cores sequentially, so parallel trials partition the same
> compute rather than adding any.

**Infrastructure as code**
> Provisioned the cloud training path with Terraform on AWS — S3 with lifecycle
> rules covering non-current versions, ECR with an image-retention policy, ECS
> Fargate, three least-privilege IAM roles with a confused-deputy guard — ran a
> real training job on it (22 s, $0.0003, cloud PR-AUC within 0.004 of local),
> and tore it down in 7.1 s against a non-empty 181 MB bucket, verified by
> querying AWS directly across ten resource classes rather than trusting
> `Destroy complete!`.

---

## Questions these bullets invite, and the honest answers

**"Where could skew still creep in?"**
Three places, all documented. The `release_after_s` timer trades online-store
staleness against tie-correctness — a tie whose halves are more than 5 wall-clock
seconds apart is miscounted, and no setting avoids both. The streaming engine
keeps state in-process, so a crash between the landing-zone write and the Redis
push leaves the store behind until state is rebuilt from Parquet. And the
on-demand feature maths exists in two forms (single-row for serving, vectorised
for training); a test asserts they agree, which is a check, not a proof.

**"Why is PR-AUC only 0.17?"**
Deliberately. Only card-velocity aggregates are used — no `V*`, `C*` or `D*`
columns, which is where most of the IEEE-CIS leaderboard signal lives. The claim
being made is about feature parity, not leaderboard rank. Against the right
baseline — amount-only at 0.0371 — it is a 4.6× lift.

**"What if `terraform apply` fails halfway?"**
Demonstrated in `terraform/README.md`, not described. A globally-taken S3 bucket
name fails at the API mid-apply, after siblings are created. State holds what
succeeded; the failed resource is absent (no half-record); re-applying converges
rather than duplicating. The one that actually bites is the fourth: a resource
created by a partial apply is still tracked, so deleting the config *and* the
state entry is what leaves something billing that nobody is watching.

**"How do you know the drift monitor works?"**
Because it is tested in both directions. Firing on shifted data proves nothing
on its own — a monitor that alerts on everything would pass that test too. The
null control is what makes the positive result mean something.

**"What was the hardest bug?"**
The partition-ordering one, because neither mechanism was wrong. Six-way
partitioning is correct. Skipping stale writes by event timestamp is correct.
Together they silently dropped writes, and the only visible symptom was a
counter that had never been wrong before. Finding it needed a check that read
values back out of Redis and compared them to the batch computation — which is
why `_assert_push_landed()` exists.
