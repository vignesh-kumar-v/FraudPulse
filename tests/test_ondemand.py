"""On-demand features must be identical row-wise and batch-wise.

A single-row serving implementation drifting from its vectorised training twin
is the quietest possible form of train/serve skew: no error, no NaN, just a
model reading a slightly different number than it was fit on.
"""

import numpy as np
import pandas as pd
import pytest

from fraudpulse.data.synthetic import make_events
from fraudpulse.features.offline import compute_offline_features
from fraudpulse.features.ondemand import compute_ondemand, compute_ondemand_frame
from fraudpulse.features.spec import FEATURE_NAMES, ONDEMAND_FEATURE_NAMES


@pytest.fixture(scope="module")
def featured() -> pd.DataFrame:
    return compute_offline_features(make_events(n_cards=40, n_events=1_000, seed=13))


def test_row_and_frame_implementations_agree(featured):
    batch = compute_ondemand_frame(featured)
    for i in range(0, len(featured), 37):
        row = featured.iloc[i]
        single = compute_ondemand(
            amount=float(row["amount"]),
            product_cd=str(row["product_cd"]),
            event_unixtime=row["event_timestamp"].timestamp(),
            stored={k: float(row[k]) for k in FEATURE_NAMES},
        )
        for name in ONDEMAND_FEATURE_NAMES:
            assert single[name] == pytest.approx(float(batch[name].iloc[i]), rel=1e-12), name


def test_no_infinities_or_nans(featured):
    out = compute_ondemand_frame(featured)
    assert np.isfinite(out.to_numpy()).all()


def test_cold_start_is_zero_not_inf():
    stored = {
        "amt_mean_24h": 0.0,
        "amt_max_7d": 0.0,
        "txn_count_7d": 0.0,
        "product_W_count_7d": 0.0,
        "last_txn_unixtime": -1.0,
    }
    out = compute_ondemand(amount=500.0, product_cd="W", event_unixtime=1_000.0, stored=stored)
    assert out["amt_over_mean_24h"] == 0.0
    assert out["amt_over_max_7d"] == 0.0
    assert out["product_freq_7d"] == 0.0
    assert out["seconds_since_last_txn"] == -1.0


def test_age_is_computed_against_the_request_clock():
    """The same stored row read later must yield a larger age."""
    stored = {"last_txn_unixtime": 1_000.0, "amt_mean_24h": 10.0, "amt_max_7d": 10.0,
              "txn_count_7d": 2.0, "product_W_count_7d": 1.0}
    early = compute_ondemand(amount=5.0, product_cd="W", event_unixtime=1_100.0, stored=stored)
    late = compute_ondemand(amount=5.0, product_cd="W", event_unixtime=5_000.0, stored=stored)
    assert early["seconds_since_last_txn"] == 100.0
    assert late["seconds_since_last_txn"] == 4_000.0
