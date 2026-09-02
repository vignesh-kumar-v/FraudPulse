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

    That lag is not free, and the naive version of it is a bug: a card's *last*
    event has no successor, so it sits in the buffer forever and never reaches
    the online store. In fraud detection that is precisely the wrong event to
    lose - the newest transaction is the one a velocity feature cares most
    about. :meth:`OnlineFeatureEngine.release_idle` fixes it on a
    **processing-time** timer: if nothing newer has arrived for a card in N
    wall-clock seconds, stop waiting and commit.

    The timer is processing-time and not event-time on purpose. The topic is
    partitioned six ways, so the *global* event clock jumps backwards on 68% of
    messages (measured; see docs/findings.md #4) and is useless as a release
    signal - it would race ahead of a lagging partition and release a buffered
    event before its tie-partner arrived, quietly reinstating the very skew the
    watermark exists to prevent. Per-card ordering, which the producer's keying
    does guarantee, is all this needs.

    The tradeoff is explicit: ``release_after_s`` bounds how stale the online
    store may be, and a tie whose halves are separated by more than that window
    will be miscounted. See docs/findings.md #2.

``features/parity.py`` measures the mismatch rate under both. See
docs/findings.md for the numbers.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from fraudpulse.features.spec import (
    FEATURE_DEFAULTS,
    FEATURE_NAMES,
    NO_LAST_TXN,
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

    def has_pending(self) -> bool:
        return bool(self._pending)

    def _flush_pending(self, up_to_exclusive: int | None = None) -> None:
        """Commit buffered events. ``None`` commits all of them."""
        if not self._pending:
            return
        if up_to_exclusive is None:
            for ts, amt, pidx in self._pending:
                self._commit(ts, amt, pidx)
            self._pending = []
            return
        keep: list[tuple[int, float, int]] = []
        for ts, amt, pidx in self._pending:
            if ts < up_to_exclusive:
                self._commit(ts, amt, pidx)
            else:
                keep.append((ts, amt, pidx))
        self._pending = keep

    @property
    def committed_ts(self) -> int | None:
        """Event time of the newest event folded into state.

        This - not any global clock - is the correct ``event_timestamp`` to
        stamp an online-store write with. See docs/findings.md #4.
        """
        return self.last_ts

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
        feats = self.snapshot()

        if tie_policy == "watermark":
            self._pending.append((ts, amount, pidx))
        else:
            self._commit(ts, amount, pidx)
        return feats

    def snapshot(self) -> dict[str, float]:
        """Current stored feature values for this card.

        Takes no timestamp on purpose. Every value here is a pure function of
        the events folded into state, so it means the same thing whenever it is
        read. Anything that depends on *now* (how long since the last
        transaction, how the current amount compares to history) is an
        on-demand feature computed at request time - see features/ondemand.py.
        """
        out: dict[str, float] = {}
        for name, w in self.windows.items():
            out[f"txn_count_{name}"] = w.count
            out[f"amt_sum_{name}"] = w.total
            out[f"amt_mean_{name}"] = w.mean
            out[f"amt_max_{name}"] = w.maximum
        out["last_txn_unixtime"] = (
            float(self.last_ts) if self.last_ts is not None else NO_LAST_TXN
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
        # card -> monotonic wall time at which its current pending event was buffered
        self._pending_since: dict[str, float] = {}
        self.max_event_ts: int = 0
        self.events_seen = 0
        self.out_of_order = 0
        self.timer_releases = 0

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
        self.max_event_ts = max(self.max_event_ts, ts)
        feats = st.features_then_update(ts, amount, product_cd, tie_policy=self.tie_policy)
        if st.has_pending():
            self._pending_since[card_id] = time.monotonic()
        else:
            self._pending_since.pop(card_id, None)
        return feats

    def release_idle(self, max_wait_s: float, now: float | None = None) -> list[str]:
        """Commit buffered events for cards idle longer than ``max_wait_s``.

        Processing-time, not event-time - see the module docstring. Returns the
        cards whose state changed so the caller can refresh them in the online
        store. Work is bounded by the number of cards currently holding a
        buffered event, not by the number of cards seen.
        """
        now = time.monotonic() if now is None else now
        released: list[str] = []
        for card_id, since in list(self._pending_since.items()):
            if now - since >= max_wait_s:
                self._states[card_id]._flush_pending(None)
                self._pending_since.pop(card_id, None)
                released.append(card_id)
        self.timer_releases += len(released)
        return released

    def finalize(self) -> list[str]:
        """End of stream: release every buffered event immediately."""
        return self.release_idle(max_wait_s=0.0)

    def peek(self, card_id: str, ts: int) -> tuple[dict[str, float], bool]:
        """Read-only feature snapshot (used by the API when Redis is cold)."""
        st = self._states.get(card_id)
        if st is None:
            return cold_start_features(), False
        st._flush_pending(ts)
        st._evict(ts)
        return st.snapshot(), True

    def pending_count(self) -> int:
        return len(self._pending_since)

    def committed_ts(self, card_id: str) -> int | None:
        st = self._states.get(card_id)
        return st.committed_ts if st else None


__all__ = [
    "AccountState",
    "OnlineFeatureEngine",
    "TiePolicy",
    "cold_start_features",
    "FEATURE_NAMES",
]
