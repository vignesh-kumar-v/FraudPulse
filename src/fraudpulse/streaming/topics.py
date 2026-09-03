"""Topic administration. Kept separate so both the CLI and tests can call it."""

from __future__ import annotations

from confluent_kafka.admin import AdminClient, NewTopic

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)


def admin() -> AdminClient:
    return AdminClient({"bootstrap.servers": settings.kafka_bootstrap})


def ensure_topic(name: str | None = None, partitions: int | None = None) -> None:
    """Create the topic if absent. Partition count is load-bearing.

    Messages are keyed by ``card_id``, so every event for a card lands on one
    partition and arrives in order. That per-card ordering is what lets the
    streaming feature engine keep a correct incremental window. More partitions
    buys parallelism; it never breaks ordering *within* a card.
    """
    name = name or settings.kafka_topic
    partitions = partitions or settings.kafka_partitions
    a = admin()
    existing = a.list_topics(timeout=10).topics
    if name in existing:
        have = len(existing[name].partitions)
        log.info("topic %s already exists with %d partition(s)", name, have)
        return
    fut = a.create_topics([NewTopic(name, num_partitions=partitions, replication_factor=1)])
    fut[name].result(timeout=20)
    log.info("created topic %s with %d partitions", name, partitions)


def delete_topic(name: str | None = None) -> None:
    name = name or settings.kafka_topic
    a = admin()
    if name not in a.list_topics(timeout=10).topics:
        return
    a.delete_topics([name])[name].result(timeout=30)
    log.info("deleted topic %s", name)


def topic_offsets(name: str | None = None) -> dict[int, int]:
    """High watermark per partition — i.e. how many messages the broker holds."""
    from confluent_kafka import Consumer, TopicPartition

    name = name or settings.kafka_topic
    c = Consumer({"bootstrap.servers": settings.kafka_bootstrap, "group.id": "fp-offsets-probe"})
    try:
        md = c.list_topics(name, timeout=10)
        out: dict[int, int] = {}
        for p in md.topics[name].partitions:
            _, high = c.get_watermark_offsets(TopicPartition(name, p), timeout=10)
            out[p] = high
        return out
    finally:
        c.close()
