"""The drift monitor has to be wrong in both directions before it is trusted."""

import numpy as np
import pandas as pd
import pytest

from fraudpulse.data.synthetic import make_events
from fraudpulse.features.offline import compute_offline_features
from fraudpulse.features.ondemand import compute_ondemand_frame
from fraudpulse.monitoring.drift import (
    KEY_FEATURES,
    MONITORED_COLUMNS,
    detect_drift,
    inject_shift,
)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    f = compute_offline_features(df)
    return pd.concat([f, compute_ondemand_frame(f)], axis=1)


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return make_events(n_cards=200, n_events=6_000, seed=101)


def test_no_alert_when_nothing_changed(base):
    """The null control. Two disjoint samples of the same process must stay quiet.

    Without this, "the monitor fired on the shifted window" proves nothing -
    a monitor that fires on everything would pass too.
    """
    a = _features(base.iloc[: len(base) // 2])
    b = _features(base.iloc[len(base) // 2 :])
    result = detect_drift(a, b, scenario="null")
    assert not result.alerted, result.summary()


def test_amount_shift_is_detected(base):
    ref = _features(base)
    cur = _features(inject_shift(base, "amount"))
    result = detect_drift(ref, cur, scenario="amount")
    assert result.alerted, result.summary()
    assert "amount" in result.drifted_columns


def test_velocity_shift_is_detected(base):
    ref = _features(base)
    cur = _features(inject_shift(base, "velocity"))
    result = detect_drift(ref, cur, scenario="velocity")
    assert result.alerted, result.summary()


def test_product_shift_is_detected(base):
    ref = _features(base)
    cur = _features(inject_shift(base, "product"))
    result = detect_drift(ref, cur, scenario="product")
    assert result.alerted, result.summary()


def test_a_single_key_feature_alerts_on_its_own(base):
    """Share thresholds average away the one-feature case; the watchlist catches it."""
    ref = _features(base)
    cur = ref.copy()
    cur["amount"] = cur["amount"] * 6.0
    result = detect_drift(ref, cur, scenario="one-column", threshold=0.99)
    assert result.drift_share < 0.99
    assert result.alerted
    assert set(result.key_features_drifted) & set(KEY_FEATURES)


def test_injection_preserves_row_count_and_cards(base):
    for kind in ("none", "amount", "velocity", "product"):
        out = inject_shift(base, kind)
        assert len(out) == len(base)
        assert set(out["card_id"]) == set(base["card_id"])


def test_velocity_injection_actually_compresses_time(base):
    out = inject_shift(base, "velocity")
    gap_before = (
        base.sort_values(["card_id", "event_timestamp"])
        .groupby("card_id")["event_timestamp"]
        .diff()
        .dt.total_seconds()
        .median()
    )
    gap_after = (
        out.sort_values(["card_id", "event_timestamp"])
        .groupby("card_id")["event_timestamp"]
        .diff()
        .dt.total_seconds()
        .median()
    )
    assert gap_after < gap_before / 5


def test_monitored_columns_are_all_numeric(base):
    f = _features(base)
    present = [c for c in MONITORED_COLUMNS if c in f.columns]
    assert len(present) == len(MONITORED_COLUMNS), "a monitored column is missing"
    assert np.isfinite(f[present].astype("float64").to_numpy()).all()
