"""The single source of truth for what a FraudPulse feature *is*.

Both computation paths import from here:

  * ``features/offline.py`` — vectorised pandas over the whole history, used to
    build the training set.
  * ``features/online.py``  — an incremental, event-at-a-time state machine used
    by the Kafka consumer to keep Redis warm.

They are deliberately *different implementations of the same spec*, not the same
code called twice. If they were the same code, the parity check in
``features/parity.py`` would be a tautology and would prove nothing about
train/serve skew. Keeping them separate is the whole point: the parity check has
to be able to fail.

Semantics that both paths must honour (these are the contract):

  1. A feature value for an event at time ``t`` is computed from that account's
     transactions in the half-open interval ``(t - W, t)`` — strictly *before*
     ``t``. The current transaction never contributes to its own aggregates.
     (Including it would leak the label-adjacent amount into the features.)
  2. Ties: transactions with the exact same timestamp as the current event are
     excluded. IEEE-CIS has plenty of these, and it is the #1 source of skew
     between a vectorised window and an arrival-ordered stream — see
     docs/findings.md.
  3. An account with no prior history gets the documented cold-start defaults
     below, never NULL. NULLs turn into silent NaNs in XGBoost and hide bugs.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Entity / timestamp
# --------------------------------------------------------------------------
ENTITY_KEY = "card_id"
EVENT_TS = "event_timestamp"

# IEEE-CIS ``TransactionDT`` is an offset in seconds from an undisclosed epoch.
# The community-accepted anchor is 2017-12-01; the absolute date does not matter
# for modelling, but a real wall-clock makes the Feast/Evidently plots readable.
DATASET_EPOCH = "2017-12-01T00:00:00"

# --------------------------------------------------------------------------
# Rolling windows
# --------------------------------------------------------------------------
WINDOWS: dict[str, int] = {
    "1h": 3_600,
    "24h": 86_400,
    "7d": 604_800,
}

# IEEE-CIS product codes. Fixed, small, and known up front, which lets us store
# per-product counters as flat columns instead of a map. That in turn lets the
# "merchant category frequency for this account" feature be an *on-demand*
# feature: it needs the current transaction's product code, which by definition
# is not in the online store yet.
PRODUCT_CODES: tuple[str, ...] = ("W", "C", "R", "H", "S")

# Cold-start defaults, applied identically by both paths.
COLD_START_SECONDS_SINCE_LAST = -1.0  # sentinel: "no prior transaction"
NO_LAST_TXN = -1.0  # sentinel for last_txn_unixtime


@dataclass(frozen=True)
class FeatureDef:
    name: str
    dtype: str  # "int64" | "float64"
    description: str
    default: float


def _build_feature_defs() -> list[FeatureDef]:
    defs: list[FeatureDef] = []
    for w in WINDOWS:
        defs.append(
            FeatureDef(
                f"txn_count_{w}",
                "int64",
                f"Transactions by this card in the {w} before the current event.",
                0,
            )
        )
    for w in WINDOWS:
        defs.append(
            FeatureDef(
                f"amt_sum_{w}",
                "float64",
                f"Sum of transaction amounts by this card in the prior {w}.",
                0.0,
            )
        )
    for w in WINDOWS:
        defs.append(
            FeatureDef(
                f"amt_mean_{w}",
                "float64",
                f"Mean transaction amount by this card in the prior {w} (0 if none).",
                0.0,
            )
        )
    for w in WINDOWS:
        defs.append(
            FeatureDef(
                f"amt_max_{w}",
                "float64",
                f"Largest transaction amount by this card in the prior {w}.",
                0.0,
            )
        )
    defs.append(
        FeatureDef(
            "last_txn_unixtime",
            "float64",
            "Unix time of the card's previous transaction, or "
            f"{NO_LAST_TXN} if it has none. Stored as an *absolute* timestamp, not "
            "as an age: an age would be correct at write time and wrong at every "
            "read after it. See docs/findings.md #3.",
            NO_LAST_TXN,
        )
    )
    defs.append(
        FeatureDef(
            "txn_count_lifetime",
            "int64",
            "All-time transaction count for this card before the current event.",
            0,
        )
    )
    for pc in PRODUCT_CODES:
        defs.append(
            FeatureDef(
                f"product_{pc}_count_7d",
                "int64",
                f"Transactions with ProductCD={pc} by this card in the prior 7d.",
                0,
            )
        )
    return defs


FEATURE_DEFS: list[FeatureDef] = _build_feature_defs()
FEATURE_NAMES: list[str] = [f.name for f in FEATURE_DEFS]
FEATURE_DTYPES: dict[str, str] = {f.name: f.dtype for f in FEATURE_DEFS}
FEATURE_DEFAULTS: dict[str, float] = {f.name: f.default for f in FEATURE_DEFS}

# --------------------------------------------------------------------------
# On-demand features: computed at request time from (stored features + the
# request payload). These are never materialised to Redis.
# --------------------------------------------------------------------------
ONDEMAND_FEATURE_NAMES: list[str] = [
    "amt_over_mean_24h",  # current amount / mean of prior 24h (velocity spike)
    "amt_over_max_7d",  # current amount / largest amount seen in prior 7d
    "product_freq_7d",  # share of this card's prior-7d txns on the same product
    # Derived from the stored absolute timestamp and the *request's* clock, so it
    # is correct whenever it is read rather than only when it was written.
    "seconds_since_last_txn",
]

# Raw request fields that go into the model alongside store features.
REQUEST_FEATURE_NAMES: list[str] = [
    "amount",
    "hour_of_day",
    "day_of_week",
]

MODEL_FEATURE_NAMES: list[str] = (
    REQUEST_FEATURE_NAMES + FEATURE_NAMES + ONDEMAND_FEATURE_NAMES
)

# Categorical raw fields kept for the model (one-hot / ordinal encoded at
# training time; the encoder is persisted with the model).
CATEGORICAL_FEATURE_NAMES: list[str] = ["product_cd", "card_network", "card_type"]

ALL_MODEL_INPUTS: list[str] = MODEL_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
