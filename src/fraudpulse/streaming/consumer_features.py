"""Feature consumer: Kafka events -> incremental features -> Feast online store.

This is the serving half of the parity story. For each event it:

  1. reads the card's current window state,
  2. emits the feature vector *as of just before* the event (never including
     the event itself - that would leak),
  3. folds the event into state,
  4. pushes the updated snapshot into Redis through Feast's push API.

Step 2 and step 4 emit *different* vectors on purpose. Step 2 is what the model
would have seen scoring this event; step 4 is what the next event for this card
will see. Conflating them is a subtle and very common source of leakage, so the
two are named apart here (``scoring_view`` vs ``store_view``).

Writes are batched: one ``push`` per event would make Feast's Redis round-trip
the bottleneck and produce a latency number that says nothing about the model.
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException
from feast.data_source import PushMode

from fraudpulse.config import settings
from fraudpulse.features.online import OnlineFeatureEngine, TiePolicy
from fraudpulse.features.spec import ENTITY_KEY, EVENT_TS, FEATURE_DTYPES, FEATURE_NAMES
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)

PUSH_SOURCE = "card_stats_push"

# How long a card's buffered event may wait for a tie-partner before we commit
# it anyway. Bounds online-store staleness; see features/online.py.
DEFAULT_RELEASE_AFTER_S = 5.0


@dataclass
class FeatureStats:
    consumed: int = 0
    pushed: int = 0
    cards: int = 0
    out_of_order: int = 0
    push_batches: int = 0
    elapsed_s: float = 0.0
    push_seconds: float = 0.0
    timer_releases: int = 0
    skipped_empty: int = 0
    scoring_rows: list[dict] = field(default_factory=list)


def _to_push_frame(rows: list[dict]) -> pd.DataFrame:
    """One row per card: the *latest* snapshot wins.

    Redis holds current state, so pushing three intermediate snapshots for the
    same card in one batch is wasted work. Deduplicating here cut push volume
    by ~35% on the full replay.
    """
    df = pd.DataFrame(rows)
    df[EVENT_TS] = pd.to_datetime(df[EVENT_TS])
    # Keep the newest row per card. Sorting first matters: pending is appended in
    # arrival order, and with six partitions interleaved that is NOT event-time
    # order, so "last in the list" is not "latest in time".
    df = df.sort_values(EVENT_TS, kind="stable").drop_duplicates(subset=[ENTITY_KEY], keep="last")
    df["created"] = pd.Timestamp.utcnow().tz_localize(None)
    for name, dtype in FEATURE_DTYPES.items():
        df[name] = df[name].astype(dtype)
    return df[[ENTITY_KEY, EVENT_TS, "created", *FEATURE_NAMES]]


def _assert_push_landed(store, frame: pd.DataFrame) -> None:
    """Read back the first batch. A push that writes nothing must not look like success.

    Costs one Redis round-trip per run and is the only thing standing between
    "the consumer logged pushed=3901" and "Redis is actually empty".
    """
    probe = str(frame[ENTITY_KEY].iloc[0])
    vals = store.get_online_features(
        features=[f"card_stats:{FEATURE_NAMES[0]}"],
        entity_rows=[{ENTITY_KEY: probe}],
    ).to_dict()
    got = vals[FEATURE_NAMES[0]][0]
    if got is None:
        raise RuntimeError(
            f"push reported success but the online store has no value for {probe!r}. "
            "Check that `to=` is a PushMode enum and that Redis is reachable."
        )
    log.info("push verified: %s -> %s=%s", probe, FEATURE_NAMES[0], got)


def run(
    *,
    topic: str | None = None,
    group_id: str | None = None,
    tie_policy: TiePolicy = "watermark",
    release_after_s: float = DEFAULT_RELEASE_AFTER_S,
    batch_size: int = 2_000,
    idle_timeout_s: float = 20.0,
    from_beginning: bool = True,
    repo_path: Path | None = None,
    capture_scoring_view: Path | None = None,
    dry_run: bool = False,
) -> FeatureStats:
    """Consume the transactions topic and keep Redis warm.

    ``capture_scoring_view`` writes every event's pre-event feature vector to
    Parquet. That file is the online side of the end-to-end parity check in
    ``scripts/verify_parity.py`` - it is what the store *actually* served,
    routed through the real broker, not a simulated replay.
    """
    from feast import FeatureStore

    topic = topic or settings.kafka_topic
    group_id = group_id or settings.kafka_group_features
    repo_path = repo_path or settings.feature_repo_dir

    store = None if dry_run else FeatureStore(repo_path=str(repo_path))
    engine = OnlineFeatureEngine(tie_policy=tie_policy)

    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "earliest" if from_beginning else "latest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 600_000,
            "fetch.min.bytes": 64 * 1024,
            "fetch.wait.max.ms": 200,
        }
    )
    consumer.subscribe([topic])

    stats = FeatureStats()
    pending: list[dict] = []
    uncommitted = 0
    stop = False

    def _handle(_sig, _frm):
        nonlocal stop
        stop = True

    prev_int = signal.signal(signal.SIGINT, _handle)
    prev_term = signal.signal(signal.SIGTERM, _handle)

    def _safe_commit() -> None:
        """Commit only when there is something to commit.

        The final flush often carries watermark-released state but no newly
        consumed messages, and committing then raises _NO_OFFSET. Swallowing
        every commit error would hide real ones, so this narrows to that case.
        """
        nonlocal uncommitted
        if uncommitted == 0:
            return
        try:
            consumer.commit(asynchronous=False)
            uncommitted = 0
        except KafkaException as exc:
            if exc.args[0].code() == KafkaError._NO_OFFSET:
                uncommitted = 0
                return
            raise

    def _stage(card: str) -> None:
        """Queue a card's current state for the online store.

        The row is stamped with **that card's own** newest committed event time.
        Stamping it with a global clock looks harmless and is not: Feast's Redis
        store skips any write whose event_timestamp is <= the one already
        stored, so a card that ever gets stamped with a fast partition's clock
        has every subsequent write silently dropped. See docs/findings.md #4.
        """
        ts = engine.committed_ts(card)
        if ts is None:
            # Nothing committed yet (first event still buffered). An all-zero
            # row would be indistinguishable from the cold-start default, so
            # there is nothing worth writing.
            stats.skipped_empty += 1
            return
        pending.append(
            {
                ENTITY_KEY: card,
                EVENT_TS: pd.Timestamp(ts, unit="s"),
                **engine.state_for(card).snapshot(),
            }
        )

    def _refresh_released() -> None:
        """Push cards whose buffered event just cleared the idle timer.

        Without this the online store permanently trails each card by its most
        recent transaction - see docs/findings.md #2.
        """
        for card in engine.release_idle(release_after_s):
            _stage(card)

    def flush() -> None:
        _refresh_released()
        if not pending:
            return
        if store is not None:
            frame = _to_push_frame(pending)
            t = time.perf_counter()
            # `to` MUST be the PushMode enum. Feast compares it with `==` against
            # PushMode.ONLINE and PushMode.ONLINE_AND_OFFLINE; PushMode is a plain
            # Enum, so a string like "online" matches neither branch and push()
            # writes nothing, raises nothing and returns None. See docs/findings.md.
            store.push(PUSH_SOURCE, frame, to=PushMode.ONLINE)
            stats.push_seconds += time.perf_counter() - t
            stats.pushed += len(frame)
            stats.push_batches += 1
            if stats.push_batches == 1:
                _assert_push_landed(store, frame)
        pending.clear()
        _safe_commit()

    t0 = time.perf_counter()
    last_msg = time.perf_counter()
    try:
        while not stop:
            msg = consumer.poll(1.0)
            if msg is None:
                if time.perf_counter() - last_msg > idle_timeout_s:
                    log.info("idle for %.0fs, stopping", idle_timeout_s)
                    break
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.warning("consumer error: %s", msg.error())
                continue

            last_msg = time.perf_counter()
            evt = json.loads(msg.value())
            ts = int(pd.Timestamp(evt["event_timestamp"]).timestamp())
            card = evt["card_id"]

            # (2) what a model scoring THIS event would have read
            scoring_view = engine.process(card, ts, float(evt["amount"]), evt["product_cd"])
            if capture_scoring_view is not None:
                stats.scoring_rows.append(
                    {
                        "transaction_id": evt["transaction_id"],
                        ENTITY_KEY: card,
                        EVENT_TS: evt["event_timestamp"],
                        **scoring_view,
                    }
                )

            # (4) what the NEXT event for this card will read
            _stage(card)

            stats.consumed += 1
            uncommitted += 1
            if len(pending) >= batch_size:
                flush()
                if stats.consumed % 50_000 == 0:
                    log.info("features: consumed=%d cards=%d", stats.consumed, len(engine))
        # Close the watermark before the final flush, otherwise every card's
        # last transaction dies in the pending buffer.
        for card in engine.finalize():
            _stage(card)
        stats.timer_releases = engine.timer_releases
        flush()
    finally:
        consumer.close()
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)

    stats.elapsed_s = time.perf_counter() - t0
    stats.cards = len(engine)
    stats.out_of_order = engine.out_of_order

    if capture_scoring_view is not None and stats.scoring_rows:
        capture_scoring_view.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(stats.scoring_rows).to_parquet(capture_scoring_view, index=False)
        log.info(
            "captured %d scoring-view rows -> %s", len(stats.scoring_rows), capture_scoring_view
        )

    log.info(
        "features done: consumed=%d pushed=%d cards=%d out_of_order=%d "
        "timer_releases=%d in %.1fs (%.0f ev/s, %.1fs in redis pushes)",
        stats.consumed,
        stats.pushed,
        stats.cards,
        stats.out_of_order,
        stats.timer_releases,
        stats.elapsed_s,
        stats.consumed / stats.elapsed_s if stats.elapsed_s else 0,
        stats.push_seconds,
    )
    if stats.out_of_order:
        log.error(
            "%d events arrived out of order - per-card partition ordering is broken, "
            "online windows for those cards are wrong",
            stats.out_of_order,
        )
    return stats
