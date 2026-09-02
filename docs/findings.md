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
