"""Landing-zone consumer: raw Kafka events -> partitioned Parquet on disk.

This is the immutable record of what actually arrived, and it is what Phase 1's
verification compares against the source CSV. It exists separately from the
feature consumer (different consumer group, same topic) for two reasons:

  1. Fan-out. Losing the raw events because the feature logic crashed would be
     unrecoverable; landing them first makes the feature path replayable.
  2. It is how the streaming feature state gets rebuilt after a restart -
     replay the last 7 days of landed Parquet through the engine.

Offsets are committed *after* the batch is durably on disk. Committing before
the write is the classic way to lose a batch to a crash and then report a
row-count mismatch you cannot explain.
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from confluent_kafka import Consumer, KafkaError

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class LandingStats:
    consumed: int = 0
    written: int = 0
    malformed: int = 0
    files: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


def _flush(batch: list[dict], out_dir: Path, seq: int) -> Path:
    df = pd.DataFrame(batch)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    path = out_dir / f"part-{seq:05d}.parquet"
    df.to_parquet(path, index=False)
    return path


def run(
    *,
    topic: str | None = None,
    group_id: str | None = None,
    out_dir: Path | None = None,
    batch_size: int = 20_000,
    idle_timeout_s: float = 20.0,
    from_beginning: bool = True,
    expect: int | None = None,
) -> LandingStats:
    """Consume until ``idle_timeout_s`` elapses with no new messages."""
    topic = topic or settings.kafka_topic
    group_id = group_id or settings.kafka_group_landing
    out_dir = out_dir or settings.landing_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    stats = LandingStats()
    batch: list[dict] = []
    seq = len(list(out_dir.glob("part-*.parquet")))
    stop = False

    def _handle(_sig, _frm):
        nonlocal stop
        stop = True
        log.info("shutdown requested, flushing...")

    prev_int = signal.signal(signal.SIGINT, _handle)
    prev_term = signal.signal(signal.SIGTERM, _handle)

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
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                log.warning("consumer error: %s", msg.error())
                continue

            last_msg = time.perf_counter()
            stats.consumed += 1
            try:
                batch.append(json.loads(msg.value()))
            except (ValueError, TypeError):
                stats.malformed += 1
                log.warning("malformed message at offset %d", msg.offset())
                continue

            if len(batch) >= batch_size:
                path = _flush(batch, out_dir, seq)
                stats.written += len(batch)
                stats.files.append(path.name)
                seq += 1
                batch.clear()
                consumer.commit(asynchronous=False)  # only after the write lands
                log.info("landed %d events (%s)", stats.written, path.name)

        if batch:
            path = _flush(batch, out_dir, seq)
            stats.written += len(batch)
            stats.files.append(path.name)
            consumer.commit(asynchronous=False)
    finally:
        consumer.close()
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)

    stats.elapsed_s = time.perf_counter() - t0
    log.info(
        "landing done: consumed=%d written=%d malformed=%d files=%d in %.1fs",
        stats.consumed,
        stats.written,
        stats.malformed,
        len(stats.files),
        stats.elapsed_s,
    )
    if expect is not None and stats.written != expect:
        log.error("ROW COUNT MISMATCH: expected %d, landed %d", expect, stats.written)
    return stats


def read_landing(out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = out_dir or settings.landing_dir
    files = sorted(out_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no landed parquet files in {out_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["event_timestamp", "transaction_id"], kind="stable").reset_index(
        drop=True
    )
