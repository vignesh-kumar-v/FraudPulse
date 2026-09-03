"""Make (entity, timestamp) unique so a point-in-time join cannot silently drop rows.

Feast's offline point-in-time join ends with

    df.drop_duplicates(join_keys + [entity_event_timestamp], keep="last")

(``feast/infra/offline_stores/dask.py``). Two transactions on the same card in
the same second therefore collapse into one, and one of them vanishes from the
training set. No warning, no row-count check, no error.

On IEEE-CIS that is 166 of 590,540 rows (0.0281%) - and those rows are **8.4%
fraud against a 3.50% base rate**, so the silent drop is 2.4x enriched in
exactly the class the model is trying to learn.

The fix is to give every transaction a distinct instant by appending a
microsecond-scale tiebreaker: the row's rank among its card's transactions that
share the second, ordered by ``transaction_id``. Both sides of the join get the
same treatment from this one function, so a feature row and its own entity row
still line up exactly, while ties become orderable instead of ambiguous.

Microseconds, not nanoseconds: Feast normalises timestamps to Arrow
``timestamp[us]``, and a nanosecond offset makes that cast raise
``ArrowInvalid: Casting from timestamp[ns] to timestamp[us] would lose data``.
A million slots inside each second is many orders of magnitude more than the
largest observed tie group (2).

Why this is safe:
  * The source data has one-second resolution, so sub-second precision is
    unused space.
  * The ordering matches the one the offline feature computation already uses
    (stable sort on ``[card_id, event_timestamp]``, which preserves
    ``transaction_id`` order), so rank k on the entity side is rank k on the
    feature side.
  * A ``<= entity_ts`` point-in-time predicate over ``T + k ns`` still selects
    the rank-k feature row and everything before it - which is the definition we
    want - instead of picking one of the tied rows arbitrarily.
"""

from __future__ import annotations

import pandas as pd

from fraudpulse.features.spec import ENTITY_KEY, EVENT_TS
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)

ORDER_COL = "transaction_id"
_MAX_TIES = 1_000_000  # one second, in microseconds


def tiebreak_timestamps(
    df: pd.DataFrame,
    *,
    entity_col: str = ENTITY_KEY,
    ts_col: str = EVENT_TS,
    order_col: str = ORDER_COL,
) -> pd.Series:
    """Return ``ts_col`` with a per-(entity, second) microsecond tiebreaker added."""
    ts = pd.to_datetime(df[ts_col])
    rank = (
        df.groupby([entity_col, ts], sort=False)[order_col].rank(method="first").astype("int64") - 1
    )
    n_tied = int((rank > 0).sum())
    if n_tied:
        log.info(
            "tiebreaking %d rows that share a (%s, %s) with another transaction",
            n_tied,
            entity_col,
            ts_col,
        )
    if rank.max() >= _MAX_TIES:
        raise ValueError(
            f"more than {_MAX_TIES} transactions share one second for a single "
            "card; the microsecond tiebreaker cannot represent that"
        )
    return ts + pd.to_timedelta(rank, unit="us")
