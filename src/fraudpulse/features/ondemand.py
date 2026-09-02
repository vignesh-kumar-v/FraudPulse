"""On-demand features: stored state x the incoming request.

These cannot be materialised into Redis, for two different reasons that are
worth keeping straight:

1. **They depend on the transaction being scored.** ``amt_over_mean_24h`` needs
   the amount of the transaction in front of you, which by definition is not in
   the store yet.
2. **They depend on when you ask.** ``seconds_since_last_txn`` is the one that
   bit us: writing an *age* into Redis produces a number that is correct for
   exactly one instant and silently wrong at every read afterwards. The store
   holds ``last_txn_unixtime`` (an absolute timestamp, stable forever) and the
   age is subtracted here, against the request's clock. See docs/findings.md #3.

Computing them in one place and calling that same function from both the
training-set builder and the FastAPI handler is the cheapest possible insurance
against skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudpulse.features.spec import (
    COLD_START_SECONDS_SINCE_LAST,
    NO_LAST_TXN,
    ONDEMAND_FEATURE_NAMES,
    PRODUCT_CODES,
)

_EPS = 1e-6
# Cap the ratios: a first-ever transaction divides by ~0 and would otherwise
# produce inf, which XGBoost tolerates but Evidently's drift stats do not.
_RATIO_CAP = 1_000.0


def compute_ondemand(
    *, amount: float, product_cd: str, event_unixtime: float, stored: dict[str, float]
) -> dict[str, float]:
    """Single-row version, used by the serving path."""
    mean_24h = float(stored.get("amt_mean_24h", 0.0))
    max_7d = float(stored.get("amt_max_7d", 0.0))
    total_7d = float(stored.get("txn_count_7d", 0.0))
    same_product = float(stored.get(f"product_{product_cd}_count_7d", 0.0))
    last_txn = float(stored.get("last_txn_unixtime", NO_LAST_TXN))

    return {
        "amt_over_mean_24h": min(amount / (mean_24h + _EPS), _RATIO_CAP) if mean_24h > 0 else 0.0,
        "amt_over_max_7d": min(amount / (max_7d + _EPS), _RATIO_CAP) if max_7d > 0 else 0.0,
        "product_freq_7d": (same_product / total_7d) if total_7d > 0 else 0.0,
        "seconds_since_last_txn": (
            float(event_unixtime - last_txn)
            if last_txn > NO_LAST_TXN
            else COLD_START_SECONDS_SINCE_LAST
        ),
    }


def compute_ondemand_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised version, used by the training-set builder.

    Kept numerically identical to :func:`compute_ondemand`; ``tests/`` asserts
    the two agree row-for-row, because "the batch version drifted from the
    serving version" is the classic way on-demand features go wrong.
    """
    amount = df["amount"].to_numpy(np.float64)
    mean_24h = df["amt_mean_24h"].to_numpy(np.float64)
    max_7d = df["amt_max_7d"].to_numpy(np.float64)
    total_7d = df["txn_count_7d"].to_numpy(np.float64)
    last_txn = df["last_txn_unixtime"].to_numpy(np.float64)
    event_unix = _event_unixtime(df)

    same_product = np.zeros(len(df), dtype=np.float64)
    product = df["product_cd"].to_numpy(object)
    for pc in PRODUCT_CODES:
        col = df[f"product_{pc}_count_7d"].to_numpy(np.float64)
        same_product = np.where(product == pc, col, same_product)

    out = pd.DataFrame(index=df.index)
    out["amt_over_mean_24h"] = np.where(
        mean_24h > 0, np.minimum(amount / (mean_24h + _EPS), _RATIO_CAP), 0.0
    )
    out["amt_over_max_7d"] = np.where(
        max_7d > 0, np.minimum(amount / (max_7d + _EPS), _RATIO_CAP), 0.0
    )
    out["product_freq_7d"] = np.where(total_7d > 0, same_product / np.maximum(total_7d, 1), 0.0)
    out["seconds_since_last_txn"] = np.where(
        last_txn > NO_LAST_TXN, event_unix - last_txn, COLD_START_SECONDS_SINCE_LAST
    )
    return out[ONDEMAND_FEATURE_NAMES]


def _event_unixtime(df: pd.DataFrame) -> np.ndarray:
    if "event_unixtime" in df.columns:
        return df["event_unixtime"].to_numpy(np.float64)
    ts = pd.to_datetime(df["event_timestamp"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    return ts.astype("datetime64[s]").astype(np.int64).to_numpy().astype(np.float64)


def calendar_features(ts: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(ts)
    return pd.DataFrame(
        {"hour_of_day": idx.hour.astype("int64"), "day_of_week": idx.dayofweek.astype("int64")},
        index=getattr(ts, "index", None),
    )
