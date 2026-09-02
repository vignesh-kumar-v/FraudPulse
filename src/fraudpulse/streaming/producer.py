"""Replay the historical dataset onto Kafka as a (sped-up) live stream.

Design notes that matter:

* **Keyed by ``card_id``.** All of a card's events therefore land on one
  partition and arrive in order, which is the precondition the incremental
  feature engine depends on. Keying by ``transaction_id`` would spread a card
  across partitions and quietly corrupt every rolling window.
* **Ordered by ``event_timestamp``.** A stream that arrives out of order is a
  different (harder) problem; this replay reproduces the ordered case and the
  consumer counts violations so the assumption stays checked rather than assumed.
* **Delivery callbacks are counted.** ``produce()`` is asynchronous and
  fire-and-forget; without tallying the callbacks you cannot honestly say "N
  events were published". Phase 1's verification is a row-count match, so the
  producer has to know its own true delivered count.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pandas as pd
from confluent_kafka import KafkaError, Producer

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger
from fraudpulse.streaming.topics import ensure_topic

log = get_logger(__name__)


@dataclass
class ProduceStats:
    attempted: int = 0
    delivered: int = 0
    failed: int = 0
    elapsed_s: float = 0.0

    @property
    def rate(self) -> float:
        return self.delivered / self.elapsed_s if self.elapsed_s else 0.0


def _serialise(row: dict) -> bytes:
    ts = row["event_timestamp"]
    payload = {
        "transaction_id": int(row["transaction_id"]),
        "card_id": str(row["card_id"]),
        "event_timestamp": (ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts)),
        "amount": float(row["amount"]),
        "product_cd": str(row["product_cd"]),
        "card_network": str(row.get("card_network", "unknown")),
        "card_type": str(row.get("card_type", "unknown")),
        "email_domain": str(row.get("email_domain", "unknown")),
        "addr1": None if pd.isna(row.get("addr1")) else float(row["addr1"]),
        "dist1": None if pd.isna(row.get("dist1")) else float(row["dist1"]),
        "is_fraud": None if row.get("is_fraud") is None else int(row["is_fraud"]),
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def replay(
    events: pd.DataFrame,
    *,
    topic: str | None = None,
    speedup: float = 0.0,
    max_events: int | None = None,
    batch_sleep_every: int = 500,
    batch_sleep_s: float = 0.02,
) -> ProduceStats:
    """Publish ``events`` to Kafka in timestamp order.

    ``speedup`` > 0 paces the replay against the data's own clock (e.g.
    ``speedup=3600`` replays one simulated hour per real second). ``speedup=0``
    goes as fast as the broker accepts, with a small periodic sleep so the
    consumer's lag is visible in the console rather than the whole file landing
    in one instant.
    """
    topic = topic or settings.kafka_topic
    ensure_topic(topic)

    if max_events:
        events = events.head(max_events)
    events = events.sort_values(["event_timestamp", "transaction_id"], kind="stable")

    stats = ProduceStats(attempted=len(events))

    def on_delivery(err: KafkaError | None, msg) -> None:
        if err is not None:
            stats.failed += 1
            log.error("delivery failed: %s", err)
        else:
            stats.delivered += 1

    producer = Producer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "linger.ms": 5,
            "batch.size": 64 * 1024,
            "compression.type": "lz4",
            # If the local queue fills we want back-pressure, not silent drops.
            "queue.buffering.max.messages": 200_000,
            "enable.idempotence": True,
        }
    )

    t0 = time.perf_counter()
    sim_start = events["event_timestamp"].iloc[0]

    for i, (_, row) in enumerate(events.iterrows()):
        if speedup > 0:
            target = (row["event_timestamp"] - sim_start).total_seconds() / speedup
            drift = target - (time.perf_counter() - t0)
            if drift > 0:
                time.sleep(drift)

        while True:
            try:
                producer.produce(
                    topic,
                    key=str(row["card_id"]).encode(),
                    value=_serialise(row),
                    callback=on_delivery,
                )
                break
            except BufferError:
                # local queue full: drain and retry rather than drop the event
                producer.poll(0.5)

        producer.poll(0)
        if speedup <= 0 and batch_sleep_every and (i + 1) % batch_sleep_every == 0:
            time.sleep(batch_sleep_s)
        if (i + 1) % 50_000 == 0:
            log.info("produced %d / %d", i + 1, stats.attempted)

    remaining = producer.flush(120)
    stats.elapsed_s = time.perf_counter() - t0
    if remaining:
        log.error("flush timed out with %d messages still queued", remaining)

    log.info(
        "produced attempted=%d delivered=%d failed=%d in %.1fs (%.0f msg/s)",
        stats.attempted,
        stats.delivered,
        stats.failed,
        stats.elapsed_s,
        stats.rate,
    )
    if stats.delivered != stats.attempted:
        log.error(
            "DELIVERY MISMATCH: attempted=%d delivered=%d - do not trust downstream counts",
            stats.attempted,
            stats.delivered,
        )
    return stats
