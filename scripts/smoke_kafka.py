#!/usr/bin/env python
"""Phase 0 verification: round-trip a message through the broker from the host.

Proves the docker-compose external listener is actually reachable, which is the
single most common way a 'Kafka is up' claim turns out to be false.
"""

from __future__ import annotations

import json
import sys
import time
import uuid

from confluent_kafka import Consumer, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger

log = get_logger("smoke")
TOPIC = "fp-smoke"


def main() -> int:
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap})
    md = admin.list_topics(timeout=10)
    log.info("connected to %s, %d broker(s)", settings.kafka_bootstrap, len(md.brokers))

    if TOPIC not in md.topics:
        fs = admin.create_topics([NewTopic(TOPIC, num_partitions=1, replication_factor=1)])
        for t, f in fs.items():
            try:
                f.result(timeout=15)
                log.info("created topic %s", t)
            except KafkaException as exc:  # already-exists is fine
                log.warning("create_topics(%s): %s", t, exc)

    token = str(uuid.uuid4())
    payload = {"token": token, "sent_at": time.time()}

    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap})
    producer.produce(TOPIC, key=b"smoke", value=json.dumps(payload).encode())
    remaining = producer.flush(15)
    if remaining:
        log.error("producer.flush left %d messages undelivered", remaining)
        return 1
    log.info("produced token=%s", token)

    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "group.id": f"smoke-{token[:8]}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([TOPIC])
    deadline = time.time() + 30
    try:
        while time.time() < deadline:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.warning("consumer error: %s", msg.error())
                continue
            got = json.loads(msg.value())
            if got.get("token") == token:
                rtt = (time.time() - got["sent_at"]) * 1000
                log.info(
                    "round-trip OK in %.1f ms  (partition=%d offset=%d)",
                    rtt,
                    msg.partition(),
                    msg.offset(),
                )
                return 0
    finally:
        consumer.close()

    log.error("did not observe token=%s within 30s", token)
    return 1


if __name__ == "__main__":
    sys.exit(main())
