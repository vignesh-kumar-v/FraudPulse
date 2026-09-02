"""The check the whole project hangs off: do the two feature paths agree?"""

import numpy as np
import pandas as pd
import pytest

from fraudpulse.data.synthetic import make_events
from fraudpulse.features.offline import compute_offline_features
from fraudpulse.features.parity import run_parity_check
from fraudpulse.features.spec import FEATURE_NAMES


def test_watermark_policy_is_exact(events):
    report = run_parity_check(events, tie_policy="watermark")
    assert report.passed, "\n" + report.summary()


def test_arrival_policy_diverges_on_duplicate_timestamps(events):
    """The naive streaming policy *must* fail, otherwise the fixture is too easy.

    If this test ever starts passing, the synthetic data has stopped producing
    duplicate timestamps and the watermark test above has become vacuous.
    """
    report = run_parity_check(events, tie_policy="arrival")
    assert not report.passed
    broken = {m.feature for m in report.mismatches}
    assert "txn_count_1h" in broken
    assert "txn_count_lifetime" in broken


def test_no_duplicate_timestamps_means_policies_agree():
    clean = make_events(n_cards=60, n_events=1_200, seed=3, duplicate_ts_rate=0.0)
    assert run_parity_check(clean, tie_policy="arrival").passed
    assert run_parity_check(clean, tie_policy="watermark").passed


def test_offline_features_are_point_in_time_correct(events):
    """Brute-force the definition on a sample and compare to the fast path."""
    feats = compute_offline_features(events)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(events), size=60, replace=False)

    for i in sample:
        row = events.iloc[i]
        hist = events[
            (events["card_id"] == row["card_id"])
            & (events["event_timestamp"] < row["event_timestamp"])
        ]
        assert feats["txn_count_lifetime"].iloc[i] == len(hist)

        for wname, wsec in [("1h", 3600), ("24h", 86400), ("7d", 604800)]:
            lo = row["event_timestamp"] - pd.Timedelta(seconds=wsec)
            win = hist[hist["event_timestamp"] > lo]
            assert feats[f"txn_count_{wname}"].iloc[i] == len(win)
            assert feats[f"amt_sum_{wname}"].iloc[i] == pytest.approx(win["amount"].sum())
            expected_max = win["amount"].max() if len(win) else 0.0
            assert feats[f"amt_max_{wname}"].iloc[i] == pytest.approx(expected_max)


def test_current_transaction_never_leaks_into_its_own_features():
    """A card's very first transaction must see an all-zero history."""
    df = make_events(n_cards=50, n_events=400, seed=5)
    feats = compute_offline_features(df)
    firsts = feats.groupby("card_id")["event_timestamp"].idxmin()
    first_rows = feats.loc[firsts]
    for w in ("1h", "24h", "7d"):
        assert (first_rows[f"txn_count_{w}"] == 0).all()
        assert (first_rows[f"amt_sum_{w}"] == 0).all()
    assert (first_rows["seconds_since_last_txn"] == -1.0).all()
    assert (first_rows["txn_count_lifetime"] == 0).all()


def test_all_declared_features_are_produced(events):
    feats = compute_offline_features(events)
    assert set(FEATURE_NAMES).issubset(feats.columns)
    assert not feats[FEATURE_NAMES].isna().any().any(), "features must never be NaN"


def test_empty_input_is_handled():
    empty = make_events(n_cards=1, n_events=1).iloc[:0]
    out = compute_offline_features(empty)
    assert len(out) == 0
