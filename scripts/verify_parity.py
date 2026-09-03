#!/usr/bin/env python
"""Phase 2 verification, the honest version.

``fraudpulse parity`` compares the two *implementations* in-process. This script
goes further and compares against what the system **actually served**:

  A. the offline path's value for each event, and
  B. the value the online path emitted for that same event after the message had
     been through a real broker, a real consumer group and real Redis.

It then does the blueprint's spot-check: pull 20 random (card, timestamp) pairs
straight out of the Feast online store and diff them against the offline table.
That last step catches a whole class of bug the in-process check cannot - type
coercion on the Redis round-trip, entity-key serialisation, TTL expiry, and a
push that reported success while writing nothing.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from fraudpulse.config import settings
from fraudpulse.data.prepare import load_events
from fraudpulse.feast_repo import get_store
from fraudpulse.features.offline import compute_offline_features
from fraudpulse.features.spec import ENTITY_KEY, EVENT_TS, FEATURE_DTYPES, FEATURE_NAMES
from fraudpulse.logging_utils import get_logger

log = get_logger("verify_parity")

RTOL, ATOL = 1e-9, 1e-6
N_SPOT_CHECKS = 20


def compare_served_vs_offline() -> dict:
    """(A) vs (B): every event that went through the broker."""
    served_path = settings.processed_dir / "online_scoring_view.parquet"
    if not served_path.exists():
        raise FileNotFoundError(
            f"{served_path} missing. Run `make features` (it captures the served view)."
        )

    events = load_events()
    offline = compute_offline_features(events).set_index("transaction_id")
    served = pd.read_parquet(served_path).set_index("transaction_id")

    common = offline.index.intersection(served.index)
    log.info("offline rows=%d served rows=%d overlap=%d", len(offline), len(served), len(common))
    if len(common) != len(offline):
        log.error(
            "COVERAGE GAP: %d events never produced an online feature vector",
            len(offline) - len(common),
        )

    off = offline.loc[common]
    onl = served.loc[common]

    per_feature = {}
    any_bad = np.zeros(len(common), dtype=bool)
    for name in FEATURE_NAMES:
        a = off[name].to_numpy(np.float64)
        b = onl[name].to_numpy(np.float64)
        bad = (
            (a != b) if FEATURE_DTYPES[name] == "int64" else ~np.isclose(a, b, rtol=RTOL, atol=ATOL)
        )
        any_bad |= bad
        if bad.any():
            per_feature[name] = {
                "n_mismatched": int(bad.sum()),
                "rate": float(bad.mean()),
                "max_abs_diff": float(np.abs(a - b).max()),
            }

    return {
        "events_compared": int(len(common)),
        "events_missing_online": int(len(offline) - len(common)),
        "rows_with_mismatch": int(any_bad.sum()),
        "row_mismatch_rate": float(any_bad.mean()) if len(common) else 0.0,
        "per_feature": per_feature,
    }


def spot_check_online_store(n: int = N_SPOT_CHECKS, seed: int = 0) -> dict:
    """The blueprint's check: N random cards, read straight out of Redis via Feast.

    Note what is being compared. Redis holds the card's *latest* state, so the
    right offline counterpart is that card's final row - the snapshot after its
    last transaction - not a random historical row. Getting this wrong produces
    a scary-looking mismatch that is entirely the checker's fault.
    """
    events = load_events()
    offline = compute_offline_features(events)

    # For each card, the state after its last event = the features that its
    # (hypothetical) next event at that same instant would read.
    last_idx = offline.groupby(ENTITY_KEY)[EVENT_TS].idxmax()
    last_rows = offline.loc[last_idx].set_index(ENTITY_KEY)

    # Cards ending on two tied transactions have no single "last offline row",
    # so folding one event into it is the wrong arithmetic. Exclude them from
    # the spot check rather than report a mismatch the system did not commit.
    at_max = events.merge(
        events.groupby(ENTITY_KEY)[EVENT_TS].max().rename("_max"),
        left_on=ENTITY_KEY,
        right_index=True,
    )
    n_at_max = at_max[at_max[EVENT_TS] == at_max["_max"]].groupby(ENTITY_KEY).size()
    eligible = last_rows.index.intersection(n_at_max[n_at_max == 1].index)
    log.info(
        "spot-check pool: %d of %d cards have an unambiguous last event",
        len(eligible),
        len(last_rows),
    )

    rng = np.random.default_rng(seed)
    cards = list(rng.choice(eligible.to_numpy(), size=min(n, len(eligible)), replace=False))

    store = get_store()
    resp = store.get_online_features(
        features=[f"card_stats:{f}" for f in FEATURE_NAMES],
        entity_rows=[{ENTITY_KEY: c} for c in cards],
    ).to_dict()
    online = pd.DataFrame(resp).set_index(ENTITY_KEY)

    rows, n_bad = [], 0
    for card in cards:
        # offline "state after the last event" = last row's features + that event
        prev = last_rows.loc[card]
        expected = _fold_event(prev)
        got = online.loc[card]
        diffs = {}
        for name in FEATURE_NAMES:
            e, g = float(expected[name]), float(got[name]) if got[name] is not None else None
            if g is None or not np.isclose(e, g, rtol=RTOL, atol=ATOL):
                diffs[name] = {"offline": e, "online": g}
        if diffs:
            n_bad += 1
        rows.append({"card_id": str(card), "n_diffs": len(diffs), "diffs": diffs})

    return {"n_checked": len(cards), "n_cards_with_diff": n_bad, "detail": rows}


def _fold_event(row: pd.Series) -> dict[str, float]:
    """Apply one event to a pre-event feature row -> the post-event state.

    Deliberately written from the spec rather than by calling either
    implementation, so this check stays independent of both.
    """
    out = {name: float(row[name]) for name in FEATURE_NAMES}
    amount = float(row["amount"])
    for w in ("1h", "24h", "7d"):
        out[f"txn_count_{w}"] += 1
        out[f"amt_sum_{w}"] += amount
        out[f"amt_mean_{w}"] = out[f"amt_sum_{w}"] / out[f"txn_count_{w}"]
        out[f"amt_max_{w}"] = max(out[f"amt_max_{w}"], amount)
    out["txn_count_lifetime"] += 1
    out["last_txn_unixtime"] = float(pd.Timestamp(row[EVENT_TS]).timestamp())
    out[f"product_{row['product_cd']}_count_7d"] += 1
    return out


def main() -> int:
    result = {
        "served_vs_offline": compare_served_vs_offline(),
        "online_store_spot_check": spot_check_online_store(),
    }
    out = settings.reports_dir / "parity_e2e.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    s = result["served_vs_offline"]
    k = result["online_store_spot_check"]
    print("\n=== served (through kafka) vs offline ===")
    print(f"events compared      : {s['events_compared']}")
    print(f"missing online       : {s['events_missing_online']}")
    print(f"rows with mismatch   : {s['rows_with_mismatch']} ({s['row_mismatch_rate']:.4%})")
    for name, d in sorted(s["per_feature"].items(), key=lambda kv: -kv[1]["n_mismatched"]):
        print(f"  {name:<26} {d['n_mismatched']:>7} ({d['rate']:.4%}) max|d|={d['max_abs_diff']:g}")
    print("\n=== redis spot check (blueprint's 20 pairs) ===")
    print(f"cards checked        : {k['n_checked']}")
    print(f"cards with any diff  : {k['n_cards_with_diff']}")
    for r in k["detail"]:
        if r["n_diffs"]:
            print(f"  {r['card_id']}: {r['diffs']}")
    print(f"\nwrote {out}")

    return 0 if (s["rows_with_mismatch"] == 0 and k["n_cards_with_diff"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
