"""Streaming-engine behaviour that the in-process parity check cannot see.

The parity check replays a whole file and compares the *scoring* view. It never
looks at what is left sitting in the online store afterwards, which is where the
watermark bug lived.
"""

import pandas as pd
import pytest

from fraudpulse.data.synthetic import make_events
from fraudpulse.features.offline import compute_offline_features
from fraudpulse.features.online import OnlineFeatureEngine
from fraudpulse.features.spec import FEATURE_NAMES


def _replay(df: pd.DataFrame, engine: OnlineFeatureEngine) -> None:
    for _, r in df.sort_values(["event_timestamp", "transaction_id"]).iterrows():
        engine.process(
            r["card_id"],
            int(r["event_timestamp"].timestamp()),
            float(r["amount"]),
            r["product_cd"],
        )


def test_watermark_buffer_would_strand_the_last_event_without_finalize():
    """Regression guard for findings.md #2.

    Every card's newest transaction is held back by the watermark. If it is
    never released, the online store trails reality by exactly the event a
    fraud model most needs.
    """
    df = make_events(n_cards=25, n_events=600, seed=17)
    engine = OnlineFeatureEngine(tie_policy="watermark")
    _replay(df, engine)

    assert engine.pending_count() > 0, "fixture should leave buffered events"

    stranded = engine.state_for(df["card_id"].iloc[-1])
    before = stranded.snapshot()["txn_count_lifetime"]
    engine.finalize()
    after = stranded.snapshot()["txn_count_lifetime"]

    assert after > before
    assert engine.pending_count() == 0


def test_finalized_state_matches_the_full_history():
    """After finalize, each card's stored state must equal its true totals."""
    df = make_events(n_cards=30, n_events=900, seed=23)
    engine = OnlineFeatureEngine(tie_policy="watermark")
    _replay(df, engine)
    engine.finalize()

    truth = df.groupby("card_id").size()
    for card, n in truth.items():
        assert engine.state_for(card).snapshot()["txn_count_lifetime"] == n


def test_release_idle_respects_the_timer():
    """A card must not be released before its idle window has elapsed."""
    df = make_events(n_cards=10, n_events=200, seed=31)
    engine = OnlineFeatureEngine(tie_policy="watermark")
    _replay(df, engine)
    buffered = engine.pending_count()
    assert buffered > 0

    assert engine.release_idle(max_wait_s=3600.0) == []
    assert engine.pending_count() == buffered

    released = engine.release_idle(max_wait_s=0.0)
    assert len(released) == buffered
    assert engine.pending_count() == 0


def test_committed_ts_is_per_card_and_monotonic():
    """findings.md #4: the online write timestamp must be the card's own clock.

    A global clock would let a fast partition stamp a slow card with a future
    timestamp, after which Feast silently drops every later write for it.
    """
    engine = OnlineFeatureEngine(tie_policy="arrival")
    engine.process("slow", 1_000, 5.0, "W")
    engine.process("fast", 9_000_000, 5.0, "W")
    assert engine.committed_ts("slow") == 1_000
    assert engine.committed_ts("fast") == 9_000_000

    engine.process("slow", 2_000, 5.0, "W")
    assert engine.committed_ts("slow") == 2_000


def test_committed_ts_is_none_before_anything_commits():
    engine = OnlineFeatureEngine(tie_policy="watermark")
    engine.process("c", 100, 1.0, "W")  # buffered, not committed
    assert engine.committed_ts("c") is None
    engine.finalize()
    assert engine.committed_ts("c") == 100


def test_arrival_policy_never_buffers():
    df = make_events(n_cards=10, n_events=200, seed=37)
    engine = OnlineFeatureEngine(tie_policy="arrival")
    _replay(df, engine)
    assert engine.pending_count() == 0


def test_stored_features_do_not_depend_on_read_time():
    """findings.md #3: nothing in the stored vector may be an age.

    Two reads of the same untouched state, notionally hours apart, must return
    identical values. ``snapshot()`` takes no clock, so this is structural - the
    test exists to stop anyone re-introducing one.
    """
    df = make_events(n_cards=8, n_events=150, seed=41)
    engine = OnlineFeatureEngine()
    _replay(df, engine)
    engine.finalize()
    card = df["card_id"].iloc[0]
    assert engine.state_for(card).snapshot() == engine.state_for(card).snapshot()


def test_out_of_order_events_are_counted():
    engine = OnlineFeatureEngine(tie_policy="arrival")
    engine.process("card_1", 2_000, 10.0, "W")
    engine.process("card_1", 1_000, 10.0, "W")
    assert engine.out_of_order == 1


def test_window_eviction_drops_old_events():
    engine = OnlineFeatureEngine(tie_policy="arrival")
    engine.process("c", 0, 100.0, "W")
    # 2h later: outside 1h, inside 24h
    feats = engine.process("c", 7_200, 50.0, "W")
    assert feats["txn_count_1h"] == 0
    assert feats["txn_count_24h"] == 1
    assert feats["amt_sum_24h"] == pytest.approx(100.0)
    assert feats["amt_max_1h"] == 0.0


def test_engine_emits_every_declared_feature():
    engine = OnlineFeatureEngine()
    feats = engine.process("c", 0, 1.0, "W")
    assert set(feats) == set(FEATURE_NAMES)


def test_offline_and_engine_agree_on_final_totals():
    """Fold each card's last event into its last offline row and compare.

    Restricted to cards whose maximum timestamp is unique. Where a card ends on
    two tied transactions the "last offline row" is ambiguous by construction,
    and folding exactly one event into it is the wrong arithmetic - which is a
    property of the check, not of the engine. test_finalized_state_matches_the_
    full_history covers the tied cards.
    """
    df = make_events(n_cards=20, n_events=500, seed=53)
    off = compute_offline_features(df)
    engine = OnlineFeatureEngine()
    _replay(df, engine)
    engine.finalize()

    tail = df.groupby("card_id")["event_timestamp"].agg(["max", "count"])
    tied = df.merge(tail, left_on="card_id", right_index=True)
    n_at_max = tied[tied["event_timestamp"] == tied["max"]].groupby("card_id").size()
    unambiguous = set(n_at_max[n_at_max == 1].index)
    assert unambiguous, "fixture must contain at least one card with a unique last event"

    last = off.loc[off.groupby("card_id")["event_timestamp"].idxmax()]
    for _, r in last[last["card_id"].isin(unambiguous)].iterrows():
        got = engine.state_for(r["card_id"]).snapshot()
        # offline row is the value *before* that event; +1 folds it in
        assert got["txn_count_lifetime"] == r["txn_count_lifetime"] + 1
        assert got["amt_max_7d"] >= r["amt_max_7d"]
