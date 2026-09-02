"""Batch (offline) feature computation — the *training* path.

Strategy: sort once by (card_id, event_timestamp), then for every event use
binary search to find the half-open index range of that card's prior
transactions inside each window, and answer the aggregates with prefix sums
(count / sum) and a sparse-table range-max (max).

This is deliberately a whole-history, index-range algorithm. The serving path
(``features/online.py``) is an arrival-ordered state machine that never sees the
future and cannot binary-search. Two different algorithms, one spec — which is
what makes ``features/parity.py`` a real test rather than a tautology.

Cost on the full IEEE-CIS train split (590k rows, ~13k cards): a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudpulse.features.spec import (
    ENTITY_KEY,
    EVENT_TS,
    FEATURE_DTYPES,
    FEATURE_NAMES,
    NO_LAST_TXN,
    PRODUCT_CODES,
    WINDOWS,
)
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)


class _SparseTableMax:
    """O(n log n) build, O(1) range-max query over a fixed float array.

    Used instead of a monotonic deque on purpose: a deque would be the same
    algorithm the streaming path uses, and reusing it would quietly make the
    parity check pass for the wrong reason.
    """

    __slots__ = ("_levels", "_log")

    def __init__(self, arr: np.ndarray) -> None:
        n = arr.size
        self._log = np.zeros(n + 1, dtype=np.int32)
        for i in range(2, n + 1):
            self._log[i] = self._log[i // 2] + 1
        levels = [arr.astype(np.float64, copy=True)]
        k = 1
        while (1 << k) <= n:
            prev = levels[-1]
            span = 1 << (k - 1)
            levels.append(np.maximum(prev[: n - (1 << k) + 1], prev[span : n - span + 1]))
            k += 1
        self._levels = levels

    def query(self, lo: int, hi: int) -> float:
        """max over [lo, hi); 0.0 for an empty range (matches the spec default)."""
        if hi <= lo:
            return 0.0
        k = int(self._log[hi - lo])
        a = self._levels[k]
        return float(max(a[lo], a[hi - (1 << k)]))


def _empty_frame(n: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in FEATURE_NAMES:
        dtype = np.int64 if FEATURE_DTYPES[name] == "int64" else np.float64
        out[name] = np.zeros(n, dtype=dtype)
    return out


def compute_offline_features(
    df: pd.DataFrame,
    *,
    entity_col: str = ENTITY_KEY,
    ts_col: str = EVENT_TS,
    amount_col: str = "amount",
    product_col: str = "product_cd",
) -> pd.DataFrame:
    """Return ``df`` (original order preserved) with the FEATURE_NAMES columns added.

    Point-in-time correct by construction: for an event at ``t`` only that
    card's transactions strictly before ``t`` are visible, and only those inside
    the window ``(t - W, t)``.
    """
    if df.empty:
        return df.assign(**{n: pd.Series(dtype=FEATURE_DTYPES[n]) for n in FEATURE_NAMES})

    required = {entity_col, ts_col, amount_col, product_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_offline_features missing columns: {sorted(missing)}")

    n = len(df)
    work = df[[entity_col, ts_col, amount_col, product_col]].copy()
    work["_orig"] = np.arange(n)
    # kind="stable" keeps the original row order within a (card, timestamp) tie,
    # which matters for reproducibility of the parity report.
    work = work.sort_values([entity_col, ts_col], kind="stable")

    ts_all = work[ts_col].to_numpy("datetime64[s]").astype(np.int64)
    amt_all = work[amount_col].to_numpy(np.float64)
    prod_all = work[product_col].to_numpy(object)
    orig_all = work["_orig"].to_numpy(np.int64)

    # group boundaries over the sorted entity column
    ent = work[entity_col].to_numpy(object)
    starts = np.flatnonzero(np.r_[True, ent[1:] != ent[:-1]])
    ends = np.r_[starts[1:], n]

    out = _empty_frame(n)
    window_items = list(WINDOWS.items())
    last_txn = np.full(n, NO_LAST_TXN, dtype=np.float64)
    lifetime = np.zeros(n, dtype=np.int64)

    prod_index = {pc: i for i, pc in enumerate(PRODUCT_CODES)}

    for s, e in zip(starts, ends, strict=True):
        ts = ts_all[s:e]
        amt = amt_all[s:e]
        rows = orig_all[s:e]
        m = ts.size

        csum = np.r_[0.0, np.cumsum(amt)]
        rmq = _SparseTableMax(amt)

        # per-product prefix counts, shape (n_products, m + 1)
        pcodes = prod_all[s:e]
        pcum = np.zeros((len(PRODUCT_CODES), m + 1), dtype=np.int64)
        for j, pc in enumerate(pcodes):
            idx = prod_index.get(pc, 0)
            pcum[:, j + 1] = pcum[:, j]
            pcum[idx, j + 1] += 1

        # hi_i = first index with ts >= t_i  -> excludes the event itself AND ties
        hi = np.searchsorted(ts, ts, side="left")
        lifetime[rows] = hi
        has_prev = hi > 0
        prev_idx = np.clip(hi - 1, 0, None)
        last_txn[rows] = np.where(has_prev, ts[prev_idx].astype(np.float64), NO_LAST_TXN)

        for wname, wsec in window_items:
            # lo_i = first index with ts > t_i - W  -> half-open (t - W, t)
            lo = np.searchsorted(ts, ts - wsec, side="right")
            cnt = (hi - lo).astype(np.int64)
            total = csum[hi] - csum[lo]
            out[f"txn_count_{wname}"][rows] = cnt
            out[f"amt_sum_{wname}"][rows] = total
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"amt_mean_{wname}"][rows] = np.where(cnt > 0, total / np.maximum(cnt, 1), 0.0)
            out[f"amt_max_{wname}"][rows] = [rmq.query(int(a), int(b)) for a, b in zip(lo, hi, strict=True)]

            if wname == "7d":
                for pc in PRODUCT_CODES:
                    j = prod_index[pc]
                    out[f"product_{pc}_count_7d"][rows] = pcum[j][hi] - pcum[j][lo]

    out["last_txn_unixtime"] = last_txn
    out["txn_count_lifetime"] = lifetime

    result = df.copy()
    for name in FEATURE_NAMES:
        result[name] = out[name]
    log.info("computed %d offline features for %d rows", len(FEATURE_NAMES), n)
    return result
