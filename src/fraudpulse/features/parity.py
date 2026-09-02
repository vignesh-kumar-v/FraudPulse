"""Offline vs. online feature parity.

Phase 2's critical check, and the reason this project exists. Two independent
implementations of ``features/spec.py`` are run over identical input and
compared value-by-value. Any disagreement is train/serve skew: the model would
be trained on one number and served another.

What this module is careful *not* to do:

  * It does not share code between the two paths (see the module docstrings in
    offline.py / online.py).
  * It does not compare with a loose tolerance and call it a pass. Integer
    counters are compared exactly; float aggregates get a relative tolerance
    sized for float64 accumulation error (1e-9), not one large enough to paper
    over a real logic difference.
  * It reports mismatches *per feature*, because "0.4% of rows differ" is
    useless and "0.4% of rows differ, all of them in txn_count_1h, all on
    duplicate timestamps" is a diagnosis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fraudpulse.features.offline import compute_offline_features
from fraudpulse.features.online import OnlineFeatureEngine, TiePolicy
from fraudpulse.features.spec import ENTITY_KEY, EVENT_TS, FEATURE_DTYPES, FEATURE_NAMES
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)

FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-6


@dataclass
class FeatureMismatch:
    feature: str
    n_mismatched: int
    n_rows: int
    max_abs_diff: float
    example_rows: list[dict] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.n_mismatched / self.n_rows if self.n_rows else 0.0


@dataclass
class ParityReport:
    tie_policy: str
    n_rows: int
    n_cards: int
    n_features: int
    rows_with_any_mismatch: int
    mismatches: list[FeatureMismatch]

    @property
    def row_mismatch_rate(self) -> float:
        return self.rows_with_any_mismatch / self.n_rows if self.n_rows else 0.0

    @property
    def passed(self) -> bool:
        return self.rows_with_any_mismatch == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["row_mismatch_rate"] = self.row_mismatch_rate
        d["passed"] = self.passed
        for m, src in zip(d["mismatches"], self.mismatches, strict=True):
            m["rate"] = src.rate
        return d

    def summary(self) -> str:
        lines = [
            f"tie_policy={self.tie_policy}  rows={self.n_rows}  cards={self.n_cards}  "
            f"features={self.n_features}",
            f"rows with >=1 mismatched feature: {self.rows_with_any_mismatch} "
            f"({self.row_mismatch_rate:.4%})",
        ]
        if not self.mismatches:
            lines.append("  all features agree exactly")
        for m in sorted(self.mismatches, key=lambda x: -x.n_mismatched):
            lines.append(
                f"  {m.feature:<26} {m.n_mismatched:>7} / {m.n_rows} "
                f"({m.rate:.4%})  max|diff|={m.max_abs_diff:g}"
            )
        return "\n".join(lines)


def replay_online(
    events: pd.DataFrame, *, tie_policy: TiePolicy = "watermark"
) -> tuple[pd.DataFrame, OnlineFeatureEngine]:
    """Feed ``events`` through the streaming engine in wall-clock order.

    Mirrors what the Kafka consumer does, minus the broker. Row order here is
    the order the producer would emit, which is also the order a card's
    partition would deliver.
    """
    ordered = events.sort_values([EVENT_TS, "transaction_id"], kind="stable")
    engine = OnlineFeatureEngine(tie_policy=tie_policy)

    ts_arr = ordered[EVENT_TS].to_numpy("datetime64[s]").astype(np.int64)
    cards = ordered[ENTITY_KEY].to_numpy(object)
    amts = ordered["amount"].to_numpy(np.float64)
    prods = ordered["product_cd"].to_numpy(object)

    rows = [
        engine.process(cards[i], int(ts_arr[i]), float(amts[i]), prods[i])
        for i in range(len(ordered))
    ]
    out = pd.DataFrame(rows, index=ordered.index)[FEATURE_NAMES]
    for name, dtype in FEATURE_DTYPES.items():
        out[name] = out[name].astype(dtype)
    return out.loc[events.index], engine


def compare(
    offline: pd.DataFrame,
    online: pd.DataFrame,
    events: pd.DataFrame,
    *,
    tie_policy: str,
    n_examples: int = 5,
) -> ParityReport:
    n = len(events)
    any_mismatch = np.zeros(n, dtype=bool)
    mismatches: list[FeatureMismatch] = []

    for name in FEATURE_NAMES:
        a = offline[name].to_numpy()
        b = online[name].to_numpy()
        if FEATURE_DTYPES[name] == "int64":
            bad = a.astype(np.int64) != b.astype(np.int64)
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        else:
            af, bf = a.astype(np.float64), b.astype(np.float64)
            bad = ~np.isclose(af, bf, rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True)
            diff = np.abs(af - bf)

        any_mismatch |= bad
        if bad.any():
            idx = np.flatnonzero(bad)[:n_examples]
            examples = [
                {
                    "transaction_id": int(events["transaction_id"].iloc[i]),
                    "card_id": str(events[ENTITY_KEY].iloc[i]),
                    "event_timestamp": str(events[EVENT_TS].iloc[i]),
                    "offline": float(a[i]),
                    "online": float(b[i]),
                }
                for i in idx
            ]
            mismatches.append(
                FeatureMismatch(
                    feature=name,
                    n_mismatched=int(bad.sum()),
                    n_rows=n,
                    max_abs_diff=float(diff.max()),
                    example_rows=examples,
                )
            )

    return ParityReport(
        tie_policy=tie_policy,
        n_rows=n,
        n_cards=int(events[ENTITY_KEY].nunique()),
        n_features=len(FEATURE_NAMES),
        rows_with_any_mismatch=int(any_mismatch.sum()),
        mismatches=mismatches,
    )


def run_parity_check(
    events: pd.DataFrame, *, tie_policy: TiePolicy = "watermark"
) -> ParityReport:
    offline = compute_offline_features(events)
    online, engine = replay_online(events, tie_policy=tie_policy)
    if engine.out_of_order:
        log.warning(
            "%d out-of-order events during replay - per-card ordering assumption violated",
            engine.out_of_order,
        )
    return compare(offline, online, events, tie_policy=tie_policy)


def write_report(reports: list[ParityReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in reports], indent=2))
    log.info("wrote parity report -> %s", path)
