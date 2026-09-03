#!/usr/bin/env python
"""Rebuild the streaming feature engine's keyed state from the landing zone.

``OnlineFeatureEngine`` holds per-card window state in process, the same way
Flink or Kafka Streams hold keyed state. That state is lost on restart. The
landing zone is the durability story: every raw event the consumer accepted was
written to Parquet *before* its offsets were committed, so the state is a pure
function of files that survived the crash.

A caveat the original docstring got wrong
-----------------------------------------
It claimed a 7-day replay was sufficient, because 7 days is the widest rolling
window. That is true for the windowed aggregates and false for the rest:

  * ``txn_count_lifetime`` is an all-time counter with no window at all.
  * ``last_txn_unixtime`` is the card's last transaction whenever it happened,
    which for a dormant card is older than any window.

A bounded replay silently under-reports both. ``--window-days`` runs the bounded
version anyway and *measures* the damage against a full replay rather than
asserting it is fine - see docs/findings.md #12.

Usage
-----
    python scripts/rebuild_state.py                  # full replay, push to Redis
    python scripts/rebuild_state.py --window-days 7  # bounded, and report the error
    python scripts/rebuild_state.py --no-push        # rebuild and verify only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraudpulse.config import settings
from fraudpulse.feast_repo import PUSH_SOURCE, get_store
from fraudpulse.features.online import OnlineFeatureEngine
from fraudpulse.features.spec import ENTITY_KEY, EVENT_TS, FEATURE_DTYPES, FEATURE_NAMES
from fraudpulse.logging_utils import get_logger
from fraudpulse.streaming.consumer_landing import read_landing

log = get_logger("rebuild")

PUSH_BATCH = 5_000


def replay(events: pd.DataFrame, *, tie_policy: str = "watermark") -> OnlineFeatureEngine:
    """Feed landed events through the engine in the order the producer emitted them."""
    ordered = events.sort_values([EVENT_TS, "transaction_id"], kind="stable")
    engine = OnlineFeatureEngine(tie_policy=tie_policy)  # type: ignore[arg-type]

    ts = ordered[EVENT_TS].to_numpy("datetime64[s]").astype(np.int64)
    cards = ordered[ENTITY_KEY].to_numpy(object)
    amts = ordered["amount"].to_numpy(np.float64)
    prods = ordered["product_cd"].to_numpy(object)

    t0 = time.perf_counter()
    for i in range(len(ordered)):
        engine.process(cards[i], int(ts[i]), float(amts[i]), prods[i])

    # Close the watermark: every buffered event is committed. Skipping this
    # leaves each card's newest transaction stranded - the bug from findings #2,
    # which recovery would otherwise reintroduce on every restart.
    engine.finalize()
    elapsed = time.perf_counter() - t0
    log.info(
        "replayed %d events over %d cards in %.1fs (%.0f ev/s)",
        len(ordered),
        len(engine),
        elapsed,
        len(ordered) / max(elapsed, 1e-9),
    )
    return engine


def snapshot_frame(engine: OnlineFeatureEngine, cards: list[str]) -> pd.DataFrame:
    """One row per card, stamped with that card's own committed event time.

    Same rule the live consumer uses. Stamping with a global clock is what made
    Feast silently drop later writes - findings #4.
    """
    rows = []
    for card in cards:
        ts = engine.committed_ts(card)
        if ts is None:
            continue
        rows.append(
            {
                ENTITY_KEY: card,
                EVENT_TS: pd.Timestamp(ts, unit="s"),
                **engine.state_for(card).snapshot(),
            }
        )
    df = pd.DataFrame(rows)
    df["created"] = pd.Timestamp.utcnow().tz_localize(None)
    for name, dtype in FEATURE_DTYPES.items():
        df[name] = df[name].astype(dtype)
    return df[[ENTITY_KEY, EVENT_TS, "created", *FEATURE_NAMES]]


def push(engine: OnlineFeatureEngine, cards: list[str]) -> int:
    from feast.data_source import PushMode

    store = get_store()
    frame = snapshot_frame(engine, cards)
    pushed = 0
    t0 = time.perf_counter()
    for start in range(0, len(frame), PUSH_BATCH):
        chunk = frame.iloc[start : start + PUSH_BATCH]
        store.push(PUSH_SOURCE, chunk, to=PushMode.ONLINE)
        pushed += len(chunk)
    log.info(
        "pushed %d card snapshots to the online store in %.1fs", pushed, time.perf_counter() - t0
    )
    return pushed


def compare_engines(
    full: OnlineFeatureEngine, bounded: OnlineFeatureEngine, cards: list[str]
) -> dict:
    """How wrong is a bounded replay? Measured per feature, not assumed."""
    per_feature: dict[str, dict] = {}
    for name in FEATURE_NAMES:
        diffs = []
        for card in cards:
            a = full.state_for(card).snapshot()[name]
            b = bounded.state_for(card).snapshot()[name]
            if a != b:
                diffs.append(abs(float(a) - float(b)))
        if diffs:
            per_feature[name] = {
                "cards_wrong": len(diffs),
                "cards_checked": len(cards),
                "rate": len(diffs) / len(cards),
                "max_abs_error": max(diffs),
                "mean_abs_error": sum(diffs) / len(diffs),
            }
    return per_feature


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--window-days",
        type=float,
        default=0.0,
        help="Replay only the last N days. 0 (default) replays everything, "
        "which is the only correct option for the unbounded counters.",
    )
    ap.add_argument("--no-push", action="store_true", help="Rebuild and verify without writing.")
    ap.add_argument(
        "--verify",
        type=int,
        default=20,
        help="Spot-check N cards against the offline path after pushing.",
    )
    ap.add_argument("--out", type=Path, default=settings.reports_dir / "state_rebuild.json")
    args = ap.parse_args()

    landed = read_landing()
    log.info(
        "landing zone: %d events, %s .. %s",
        len(landed),
        landed[EVENT_TS].min(),
        landed[EVENT_TS].max(),
    )

    engine = replay(landed)
    cards = sorted({str(c) for c in landed[ENTITY_KEY].unique()})

    result: dict = {
        "landed_events": int(len(landed)),
        "cards": len(cards),
        "window_days": args.window_days or "full",
        "pending_after_finalize": engine.pending_count(),
        "out_of_order": engine.out_of_order,
    }

    # -- the bounded-replay measurement ------------------------------------
    if args.window_days:
        cutoff = landed[EVENT_TS].max() - pd.Timedelta(days=args.window_days)
        window = landed[landed[EVENT_TS] > cutoff]
        log.info("bounded replay: %d of %d events after %s", len(window), len(landed), cutoff)
        bounded = replay(window)
        shared = sorted(set(cards) & {str(c) for c in window[ENTITY_KEY].unique()})
        damage = compare_engines(engine, bounded, shared)
        result["bounded_replay"] = {
            "events_replayed": int(len(window)),
            "cards_present": len(shared),
            "features_wrong": damage,
        }
        log.warning(
            "bounded replay is wrong on %d of %d features:", len(damage), len(FEATURE_NAMES)
        )
        for name, d in sorted(damage.items(), key=lambda kv: -kv[1]["rate"]):
            log.warning(
                "  %-24s %5.1f%% of cards, max error %.4g",
                name,
                d["rate"] * 100,
                d["max_abs_error"],
            )
        engine = bounded  # honour the flag: push what the bounded replay produced

    # -- write ---------------------------------------------------------------
    if not args.no_push:
        result["pushed"] = push(engine, cards)

        if args.verify:
            from verify_parity import spot_check_online_store

            check = spot_check_online_store(n=args.verify)
            result["spot_check"] = {
                "n_checked": check["n_checked"],
                "n_cards_with_diff": check["n_cards_with_diff"],
            }
            log.info(
                "post-rebuild spot check: %d/%d cards exact",
                check["n_checked"] - check["n_cards_with_diff"],
                check["n_checked"],
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    log.info("wrote %s", args.out)

    sc = result.get("spot_check")
    if sc and sc["n_cards_with_diff"]:
        log.error(
            "rebuilt state does not match the offline path for %d cards", sc["n_cards_with_diff"]
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
