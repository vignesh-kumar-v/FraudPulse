"""On-demand features: stored state x the incoming request.

These can never be materialised into Redis, because they depend on a field of
the transaction being scored (its amount, its product code). Computing them in
one place and calling that same function from both the training-set builder and
the FastAPI handler is the cheapest possible insurance against skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudpulse.features.spec import ONDEMAND_FEATURE_NAMES, PRODUCT_CODES

_EPS = 1e-6
# Cap the ratios: a first-ever transaction divides by ~0 and would otherwise
# produce inf, which XGBoost tolerates but Evidently's drift stats do not.
_RATIO_CAP = 1_000.0


def compute_ondemand(
    *, amount: float, product_cd: str, stored: dict[str, float]
) -> dict[str, float]:
    """Single-row version, used by the serving path."""
    mean_24h = float(stored.get("amt_mean_24h", 0.0))
    max_7d = float(stored.get("amt_max_7d", 0.0))
    total_7d = float(stored.get("txn_count_7d", 0.0))
    same_product = float(stored.get(f"product_{product_cd}_count_7d", 0.0))

    return {
        "amt_over_mean_24h": min(amount / (mean_24h + _EPS), _RATIO_CAP) if mean_24h > 0 else 0.0,
        "amt_over_max_7d": min(amount / (max_7d + _EPS), _RATIO_CAP) if max_7d > 0 else 0.0,
        "product_freq_7d": (same_product / total_7d) if total_7d > 0 else 0.0,
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
    return out[ONDEMAND_FEATURE_NAMES]


def calendar_features(ts: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(ts)
    return pd.DataFrame(
        {"hour_of_day": idx.hour.astype("int64"), "day_of_week": idx.dayofweek.astype("int64")},
        index=getattr(ts, "index", None),
    )
