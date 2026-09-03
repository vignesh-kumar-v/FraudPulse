"""Synthetic event generator — for tests and for the drift-injection experiment.

Two uses:
  * ``make test`` must run on a fresh clone with no Kaggle credentials, so the
    parity and feature unit tests build their fixtures here.
  * Phase 4 needs a stream whose distribution provably shifted; ``shift=`` gives
    a controlled way to produce one.

Deliberately includes the awkward cases that break naive implementations:
duplicate timestamps within a card, long idle gaps, single-transaction cards,
and bursts tight enough to sit inside the 1h window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudpulse.features.spec import DATASET_EPOCH, PRODUCT_CODES


def make_events(
    n_cards: int = 200,
    n_events: int = 5_000,
    *,
    seed: int = 7,
    duplicate_ts_rate: float = 0.08,
    shift: str | None = None,
) -> pd.DataFrame:
    """Generate a canonical-schema event frame.

    ``shift`` selects a distribution shift for the drift test:
      ``"amount"``  - 4x heavier transaction amounts
      ``"velocity"``- transactions bunched ~20x tighter in time
      ``"product"`` - product mix collapses onto a single category
    """
    rng = np.random.default_rng(seed)
    epoch = pd.Timestamp(DATASET_EPOCH)

    cards = np.array([f"card_{i:05d}" for i in range(n_cards)])
    # Zipf-ish activity: a few very busy cards, a long tail of one-offs.
    weights = 1.0 / np.arange(1, n_cards + 1) ** 0.8
    weights /= weights.sum()
    owner = rng.choice(cards, size=n_events, p=weights)

    gap_scale = 900.0 if shift == "velocity" else 18_000.0
    gaps = rng.exponential(gap_scale, size=n_events)
    offsets = np.cumsum(gaps).astype(np.int64)

    # force exact-duplicate timestamps: the single nastiest case for a
    # streaming window, and the one the parity check is built to catch
    n_dup = int(n_events * duplicate_ts_rate)
    if n_dup:
        idx = rng.choice(np.arange(1, n_events), size=n_dup, replace=False)
        offsets[idx] = offsets[idx - 1]

    amt_scale = 4.0 if shift == "amount" else 1.0
    amount = np.round(rng.lognormal(mean=3.4, sigma=1.1, size=n_events) * amt_scale, 2)

    if shift == "product":
        product = np.full(n_events, "C", dtype=object)
    else:
        product = rng.choice(list(PRODUCT_CODES), size=n_events, p=[0.7, 0.12, 0.08, 0.06, 0.04])

    # Fraud correlates with velocity + amount so a model can actually learn it.
    base = 0.01 + 0.05 * (amount > np.quantile(amount, 0.97))
    is_fraud = (rng.random(n_events) < base).astype("int8")

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, n_events + 1, dtype="int64"),
            "card_id": owner,
            "event_timestamp": epoch + pd.to_timedelta(offsets, unit="s"),
            "amount": amount,
            "product_cd": product,
            "card_network": rng.choice(["visa", "mastercard", "discover"], n_events),
            "card_type": rng.choice(["credit", "debit"], n_events),
            "email_domain": "example.com",
            "addr1": rng.integers(100, 500, n_events).astype("float64"),
            "dist1": rng.integers(0, 200, n_events).astype("float64"),
            "is_fraud": is_fraud,
        }
    )
    return df.sort_values(["event_timestamp", "transaction_id"], kind="stable").reset_index(
        drop=True
    )
