"""Streaming (online) feature computation — the *serving* path.

An arrival-ordered, keyed state machine: one :class:`AccountState` per card,
holding only the last 7 days of that card's events. No lookahead, no sort, no
binary search — exactly the constraints a real stream processor works under.
This is the deliberate counterpart to the vectorised whole-history algorithm in
``features/offline.py``.

Tie policy
----------
The spec says an event at ``t`` must not see other events at exactly ``t``.
Offline that is one ``searchsorted(..., side="left")``. Online it is genuinely
hard, because by the time the second same-timestamp event arrives the first one
has already been folded into the running aggregates.

Two policies are implemented so the cost of the choice is measurable rather than
assumed:

``"arrival"``
    Naive. Fold every event into state as it arrives. Fast, but an event sees
    earlier events that share its timestamp -> skew vs. offline.
``"watermark"`` (default)
    Hold events at the current maximum timestamp in a pending buffer and only
    fold them into the window state once an event with a strictly larger
    timestamp arrives. Matches the offline semantics; costs one extra buffer and
    means state lags the stream by one timestamp tick.

``features/parity.py`` measures the mismatch rate under both. See
docs/findings.md for the numbers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from fraudpulse.features.spec import (
    COLD_START_SECONDS_SINCE_LAST,
    FEATURE_DEFAULTS,
    FEATURE_NAMES,
    PRODUCT_CODES,
    WINDOWS,
)

TiePolicy = Literal["arrival", "watermark"]

_MAX_WINDOW = max(WINDOWS.values())
_PRODUCT_INDEX = {pc: i for i, pc in enumerate(PRODUCT_CODES)}


@dataclass
class _WindowState:
    """Running count/sum/max over a time window, maintained incrementally."""

    seconds: int
    events: deque[tuple[int, float]] = field(default_factory=deque)  # (ts, amount)
    max_dq: deque[tuple[int, float]] = field(default_factory=deque)  # monotonic decreasing
    count: int = 0
    total: float = 0.0

    def evict(self, now: int) -> None:
        cutoff = now - self.seconds  # keep strictly greater than cutoff
        ev = self.events
        while ev and ev[0][0] <= cutoff:
            _, amt = ev.popleft()
            self.count -= 1
            self.total -= amt
        mq = self.max_dq
        while mq and mq[0][0] <= cutoff:
            mq.popleft()
        if self.count == 0:
            # Re-zero rather than let float error accumulate forever. Without
            # this, total drifts to ~1e-12 on an empty window and the parity
            # check flags a "mismatch" that is pure floating-point residue.
            self.total = 0.0

    def add(self, ts: int, amount: float) -> None:
        self.events.append((ts, amount))
        self.count += 1
        self.total += amount
        mq = self.max_dq
        while mq and mq[-1][1] <= amount:
            mq.pop()
        mq.append((ts, amount))

    @property
    def maximum(self) -> float:
        return self.max_dq[0][1] if self.max_dq else 0.0

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class AccountState:
    """Per-card keyed state. Bounded by the 7d retention window."""

    __slots__ = ("windows", "product_events", "product_counts", "last_ts", "lifetime", "_pending")

    def __init__(self) -> None:
        self.windows: dict[str, _WindowState] = {
            name: _WindowState(seconds=sec) for name, sec in WINDOWS.items()
        }
        # per-product counters, retained over the 7d window
        self.product_events: deque[tuple[int, int]] = deque()  # (ts, product_index)
        self.product_counts: list[int] = [0] * len(PRODUCT_CODES)
        self.last_ts: int | None = None
        self.lifetime: int = 0
        self._pending: list[tuple[int, float, int]] = []

    # -- internals ---------------------------------------------------------
    def _evict(self, now: int) -> None:
        for w in self.windows.values():
            w.evict(now)
        cutoff = now - _MAX_WINDOW
        pe = self.product_events
        while pe and pe[0][0] <= cutoff:
            _, idx = pe.popleft()
            self.product_counts[idx] -= 1

    def _commit(self, ts: int, amount: float, product_idx: int) -> None:
        for w in self.windows.values():
            w.add(ts, amount)
        self.product_events.append((ts, product_idx))
        self.product_counts[product_idx] += 1
        self.lifetime += 1
        self.last_ts = ts if self.last_ts is None else max(self.last_ts, ts)

    def _flush_pending(self, up_to_exclusive: int) -> None:
        if not self._pending:
            return
        keep: list[tuple[int, float, int]] = []
        for ts, amt, pidx in self._pending:
            if ts < up_to_exclusive:
                self._commit(ts, amt, pidx)
            else:
                keep.append((ts, amt, pidx))
        self._pending = keep

    # -- public ------------------------------------------------------------
    def features_then_update(
        self,
        ts: int,
        amount: float,
        product_cd: str,
        *,
        tie_policy: TiePolicy = "watermark",
    ) -> dict[str, float]:
        """Emit the feature vector *as of just before* this event, then absorb it."""
        pidx = _PRODUCT_INDEX.get(product_cd, 0)

        if tie_policy == "watermark":
            # everything strictly earlier than this event is now safe to commit
            self._flush_pending(ts)

        self._evict(ts)
        feats = self.snapshot(ts)

        if tie_policy == "watermark":
            self._pending.append((ts, amount, pidx))
        else:
            self._commit(ts, amount, pidx)
        return feats

    def snapshot(self, ts: int) -> dict[str, float]:
        """Current feature values for an event happening at ``ts``."""
        out: dict[str, float] = {}
        for name, w in self.windows.items():
            out[f"txn_count_{name}"] = w.count
            out[f"amt_sum_{name}"] = w.total
            out[f"amt_mean_{name}"] = w.mean
            out[f"amt_max_{name}"] = w.maximum
        out["seconds_since_last_txn"] = (
            float(ts - self.last_ts) if self.last_ts is not None else COLD_START_SECONDS_SINCE_LAST
        )
        out["txn_count_lifetime"] = self.lifetime
        for pc, idx in _PRODUCT_INDEX.items():
            out[f"product_{pc}_count_7d"] = self.product_counts[idx]
        return out


def cold_start_features() -> dict[str, float]:
    """Feature vector for a card the online store has never seen."""
    return dict(FEATURE_DEFAULTS)


class OnlineFeatureEngine:
    """Owns keyed state for every card seen on the stream.

    Deliberately in-process (the same model Flink/Kafka-Streams use for keyed
    state). Durability comes from the landing zone: ``scripts/rebuild_state.py``
    replays the last 7 days of landed Parquet to reconstruct this on restart.
    """

    def __init__(self, tie_policy: TiePolicy = "watermark") -> None:
        self.tie_policy: TiePolicy = tie_policy
        self._states: dict[str, AccountState] = {}
        self.events_seen = 0
        self.out_of_order = 0

    def __len__(self) -> int:
        return len(self._states)

    def state_for(self, card_id: str) -> AccountState:
        st = self._states.get(card_id)
        if st is None:
            st = AccountState()
            self._states[card_id] = st
        return st

    def process(self, card_id: str, ts: int, amount: float, product_cd: str) -> dict[str, float]:
        st = self.state_for(card_id)
        if st.last_ts is not None and ts < st.last_ts:
            # Per-card ordering is guaranteed only because the producer keys
            # messages by card_id, so all of a card's events land on one
            # partition. Count violations anyway - if this is ever non-zero the
            # partitioning assumption has broken.
            self.out_of_order += 1
        self.events_seen += 1
        return st.features_then_update(ts, amount, product_cd, tie_policy=self.tie_policy)

    def peek(self, card_id: str, ts: int) -> tuple[dict[str, float], bool]:
        """Read-only feature snapshot (used by the API when Redis is cold)."""
        st = self._states.get(card_id)
        if st is None:
            return cold_start_features(), False
        st._flush_pending(ts)
        st._evict(ts)
        return st.snapshot(ts), True


__all__ = [
    "AccountState",
    "OnlineFeatureEngine",
    "TiePolicy",
    "cold_start_features",
    "FEATURE_NAMES",
]
