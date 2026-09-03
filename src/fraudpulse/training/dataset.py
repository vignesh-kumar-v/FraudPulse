"""Build the training set through Feast, not around it.

The tempting shortcut is to join the offline feature parquet to the labels with
a pandas merge and move on. That would work, and it would also quietly delete
the only guarantee this project exists to make: that training reads features
through the *same registry* the serving path reads them through.

So the training set comes from ``store.get_historical_features`` against the
``fraud_serving_v1`` feature service. If a feature is renamed, retyped or
dropped in ``feature_repo/definitions.py``, training breaks - which is the
correct outcome. A pandas merge would keep working and start serving a model
fitted on columns that no longer exist.

The time split is chronological, never random. Fraud data has strong temporal
structure (campaigns, compromised BINs, seasonality); a random split lets the
model see the future for a card whose other transactions are in train, and
inflates PR-AUC by a margin that is entirely fictional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudpulse.config import settings
from fraudpulse.feast_repo import FEATURE_SERVICE, get_store
from fraudpulse.features.ondemand import calendar_features
from fraudpulse.features.spec import (
    ALL_MODEL_INPUTS,
    CATEGORICAL_FEATURE_NAMES,
    ENTITY_KEY,
    EVENT_TS,
    MODEL_FEATURE_NAMES,
)
from fraudpulse.features.timeline import tiebreak_timestamps
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)

LABEL = "is_fraud"
TRAINING_PARQUET = "training_set.parquet"


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    boundaries: tuple[pd.Timestamp, pd.Timestamp]
    # The exact string -> integer map used to encode the categoricals. Persisted
    # with the model; re-deriving it at serving time is how `visa` ends up
    # encoded as 2 in production and 0 in training.
    categories: dict[str, list[str]]

    def describe(self) -> dict:
        return {
            "n_train": len(self.X_train),
            "n_valid": len(self.X_valid),
            "n_test": len(self.X_test),
            "fraud_rate_train": float(self.y_train.mean()),
            "fraud_rate_valid": float(self.y_valid.mean()),
            "fraud_rate_test": float(self.y_test.mean()),
            "split_at": [str(b) for b in self.boundaries],
        }


def build_training_set(events: pd.DataFrame, *, use_feast: bool = True) -> pd.DataFrame:
    """Point-in-time join of labels to features, via the Feast feature service."""
    entity_df = events[
        [
            "transaction_id",
            ENTITY_KEY,
            EVENT_TS,
            "amount",
            "product_cd",
            "card_network",
            "card_type",
            "email_domain",
            "addr1",
            "dist1",
            LABEL,
        ]
    ].copy()
    # Without this, Feast's point-in-time join drops one of every pair of
    # same-card same-second transactions - 166 rows on IEEE-CIS, 8.4% of them
    # fraud against a 3.50% base rate. See features/timeline.py.
    entity_df[EVENT_TS] = tiebreak_timestamps(entity_df)
    entity_df["event_unixtime"] = (
        entity_df[EVENT_TS].astype("datetime64[s]").astype(np.int64).astype(np.float64)
    )

    if use_feast:
        store = get_store()
        svc = store.get_feature_service(FEATURE_SERVICE)
        log.info(
            "get_historical_features via feature service '%s' (%d rows)",
            FEATURE_SERVICE,
            len(entity_df),
        )
        df = store.get_historical_features(entity_df=entity_df, features=svc).to_df()
    else:
        # Escape hatch for CI, where there is no registry. Documented as a
        # fallback so nobody mistakes it for the normal path.
        from fraudpulse.features.offline import compute_offline_features
        from fraudpulse.features.ondemand import compute_ondemand_frame

        log.warning("use_feast=False: bypassing the registry, features come from pandas")
        feats = compute_offline_features(events)
        df = entity_df.merge(
            feats.drop(
                columns=[c for c in entity_df.columns if c != "transaction_id"], errors="ignore"
            ),
            on="transaction_id",
            how="left",
        )
        df = pd.concat([df, compute_ondemand_frame(df)], axis=1)

    cal = calendar_features(df[EVENT_TS])
    df = pd.concat([df.reset_index(drop=True), cal.reset_index(drop=True)], axis=1)

    missing = [c for c in ALL_MODEL_INPUTS if c not in df.columns]
    if missing:
        raise KeyError(f"feature service did not return: {missing}")

    out = df[["transaction_id", ENTITY_KEY, EVENT_TS, LABEL, *ALL_MODEL_INPUTS]]
    out = out.sort_values(EVENT_TS, kind="stable").reset_index(drop=True)

    # Row-count gate. The point-in-time join has a documented habit of losing
    # rows quietly (features/timeline.py); an assertion here is the difference
    # between finding that in five minutes and finding it never.
    if len(out) != len(events):
        lost = set(events["transaction_id"]) - set(out["transaction_id"])
        raise RuntimeError(
            f"point-in-time join lost {len(events) - len(out)} of {len(events)} rows "
            f"(e.g. transaction_ids {sorted(lost)[:5]}). Refusing to train on a "
            "silently truncated dataset."
        )

    log.info(
        "training set: %d rows, %d features, fraud_rate=%.4f",
        len(out),
        len(ALL_MODEL_INPUTS),
        out[LABEL].mean(),
    )
    return out


def encode_categoricals(df: pd.DataFrame, categories: dict[str, list[str]] | None = None):
    """Ordinal-encode the categorical columns; return (frame, category map).

    The map is persisted with the model. Re-deriving categories at serving time
    from whatever happens to be in the request is how a serving path ends up
    encoding `visa` as 2 when training encoded it as 0.
    """
    out = df.copy()
    cats = categories or {}
    for col in CATEGORICAL_FEATURE_NAMES:
        if col not in out.columns:
            continue
        if col not in cats:
            cats[col] = sorted(out[col].astype(str).unique().tolist())
        lookup = {v: i for i, v in enumerate(cats[col])}
        # -1 for unseen values: an explicit "I have not seen this" bucket beats
        # silently folding it into whatever category happens to be index 0.
        out[col] = out[col].astype(str).map(lookup).fillna(-1).astype("int32")
    return out, cats


def chronological_split(
    df: pd.DataFrame, *, valid_frac: float = 0.15, test_frac: float = 0.15
) -> Split:
    df = df.sort_values(EVENT_TS, kind="stable").reset_index(drop=True)
    n = len(df)
    i_valid = int(n * (1 - valid_frac - test_frac))
    i_test = int(n * (1 - test_frac))
    b = (df[EVENT_TS].iloc[i_valid], df[EVENT_TS].iloc[i_test])

    # Categories are derived from the TRAIN slice only. Deriving them from the
    # whole frame would leak the existence of test-only category values into the
    # encoding, and would also disagree with what serving can know.
    _, categories = encode_categoricals(df.iloc[:i_valid])
    encoded, _ = encode_categoricals(df, categories)
    X = encoded[ALL_MODEL_INPUTS]
    y = encoded[LABEL].astype(int)

    split = Split(
        X_train=X.iloc[:i_valid],
        y_train=y.iloc[:i_valid],
        X_valid=X.iloc[i_valid:i_test],
        y_valid=y.iloc[i_valid:i_test],
        X_test=X.iloc[i_test:],
        y_test=y.iloc[i_test:],
        boundaries=b,
        categories=categories,
    )
    log.info("chronological split: %s", split.describe())
    return split


def load_or_build(events: pd.DataFrame | None = None, *, rebuild: bool = False) -> pd.DataFrame:
    path = settings.processed_dir / TRAINING_PARQUET
    if path.exists() and not rebuild:
        log.info("reusing %s", path)
        return pd.read_parquet(path)
    if events is None:
        from fraudpulse.data.prepare import load_events

        events = load_events()
    df = build_training_set(events)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


__all__ = [
    "LABEL",
    "MODEL_FEATURE_NAMES",
    "Split",
    "build_training_set",
    "chronological_split",
    "encode_categoricals",
    "load_or_build",
]
