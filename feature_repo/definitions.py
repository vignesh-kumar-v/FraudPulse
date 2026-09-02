"""Feast feature definitions for FraudPulse.

Read this alongside ``src/fraudpulse/features/spec.py``: the Python spec is what
the two computation paths implement, and this file is how Feast learns about it.
The schema below is *generated* from the spec, so a feature cannot exist in one
place and not the other.

Three source types are in play, and the distinction is the whole train/serve
story:

``card_stats_batch`` (FileSource)
    Parquet written by the offline path. Feeds ``get_historical_features`` for
    point-in-time-correct training data.

``card_stats_push`` (PushSource)
    How the streaming consumer writes the *same* feature names into Redis. The
    PushSource wraps the batch source, so Feast treats them as one feature view
    with two write paths - which is exactly what we want to be able to claim.

``transaction_request`` (RequestSource)
    Fields that only exist at request time: the amount being scored, its product
    code, and the request's clock. These drive the on-demand feature view below,
    which can never be materialised because it depends on the transaction being
    scored and on when the question is asked.
"""

from datetime import timedelta

import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
    ValueType,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64, String

from fraudpulse.features.spec import (
    ENTITY_KEY,
    FEATURE_DEFS,
    ONDEMAND_FEATURE_NAMES,
    PRODUCT_CODES,
)

_FEAST_TYPES = {"int64": Int64, "float64": Float64}

card = Entity(
    name=ENTITY_KEY,
    join_keys=[ENTITY_KEY],
    value_type=ValueType.STRING,
    description="Payment card / account. Derived from IEEE-CIS card1.",
)

card_stats_batch = FileSource(
    name="card_stats_batch",
    path="data/card_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
    description="Offline feature values produced by fraudpulse.features.offline.",
)

card_stats_push = PushSource(
    name="card_stats_push",
    batch_source=card_stats_batch,
    description="Streaming write path: the Kafka consumer pushes here, Feast fans "
    "it out to Redis (online) and optionally back to Parquet (offline).",
)

card_stats = FeatureView(
    name="card_stats",
    entities=[card],
    # TTL bounds how stale an online value may be before Feast stops serving it.
    # Sized just past the widest window (7d) so a card that goes quiet for over a
    # week reads as cold-start rather than silently serving week-old aggregates.
    ttl=timedelta(days=8),
    schema=[Field(name=f.name, dtype=_FEAST_TYPES[f.dtype], description=f.description)
            for f in FEATURE_DEFS],
    online=True,
    source=card_stats_push,
    tags={"team": "fraud", "phase": "2"},
)

transaction_request = RequestSource(
    name="transaction_request",
    schema=[
        Field(name="amount", dtype=Float64),
        Field(name="product_cd", dtype=String),
        # The request's own clock. Required because `seconds_since_last_txn` is
        # derived here rather than stored: the store holds an absolute
        # `last_txn_unixtime`, and the age is only meaningful relative to when
        # you ask. See docs/findings.md #3.
        Field(name="event_unixtime", dtype=Float64),
    ],
)


@on_demand_feature_view(
    sources=[card_stats, transaction_request],
    schema=[Field(name=n, dtype=Float64) for n in ONDEMAND_FEATURE_NAMES],
    description="Ratios of the transaction being scored against the card's own "
    "history. Cannot be materialised: they depend on the request.",
)
def txn_ratios(inputs: pd.DataFrame) -> pd.DataFrame:
    # Imported lazily so Feast's registry parse does not need the whole package
    # graph, and so there is exactly one implementation of this maths.
    from fraudpulse.features.ondemand import compute_ondemand_frame

    return compute_ondemand_frame(inputs)


fraud_serving_v1 = FeatureService(
    name="fraud_serving_v1",
    features=[card_stats, txn_ratios],
    description="Everything the fraud model reads from the store, versioned as a "
    "unit. Training and serving both request this service by name so they "
    "cannot drift apart.",
    tags={"model": "fraudpulse-fraud-classifier"},
)

__all__ = [
    "card",
    "card_stats",
    "card_stats_batch",
    "card_stats_push",
    "fraud_serving_v1",
    "transaction_request",
    "txn_ratios",
    "PRODUCT_CODES",
]
