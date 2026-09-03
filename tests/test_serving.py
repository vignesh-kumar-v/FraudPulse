"""Serving-path tests that need no Redis, no MLflow and no model.

These cover the encoding contract, which is where serving skew actually lives.
The end-to-end path is covered by scripts/verify_parity.py against the real
stack; what a unit test can add is proof that the *encoder* cannot silently
reshape or mis-map its input.
"""

import numpy as np
import pandas as pd
import pytest

from fraudpulse.features.spec import ALL_MODEL_INPUTS, CATEGORICAL_FEATURE_NAMES
from fraudpulse.schema import ScoreRequest, TransactionEvent
from fraudpulse.serving.model_store import LoadedModel


class _StubModel:
    """Returns the feature values it was handed, so tests can assert on them."""

    def __init__(self):
        self.last: pd.DataFrame | None = None

    def predict_proba(self, X):
        self.last = X
        return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.42)])


@pytest.fixture
def loaded() -> LoadedModel:
    return LoadedModel(
        model=_StubModel(),
        version="1",
        feature_order=list(ALL_MODEL_INPUTS),
        categories={
            "product_cd": ["C", "H", "R", "S", "W"],
            "card_network": ["amex", "discover", "mastercard", "visa"],
            "card_type": ["credit", "debit"],
            "email_domain": ["gmail.com", "yahoo.com"],
        },
        model_type="xgboost",
        threshold=0.5,
    )


def test_encode_preserves_column_order(loaded):
    X = loaded.encode(dict.fromkeys(ALL_MODEL_INPUTS, 1.0))
    assert list(X.columns) == loaded.feature_order


def test_categories_map_the_same_way_training_did(loaded):
    row = dict.fromkeys(ALL_MODEL_INPUTS, 0.0)
    row.update(product_cd="W", card_network="visa", card_type="debit", email_domain="yahoo.com")
    X = loaded.encode(row)
    assert X["product_cd"].iloc[0] == 4  # index of "W"
    assert X["card_network"].iloc[0] == 3  # index of "visa"
    assert X["card_type"].iloc[0] == 1
    assert X["email_domain"].iloc[0] == 1


def test_unseen_category_becomes_minus_one_not_zero(loaded):
    """-1 is an explicit "never seen"; 0 would silently alias to a real category."""
    row = dict.fromkeys(ALL_MODEL_INPUTS, 0.0)
    row["card_network"] = "a-network-that-did-not-exist-at-training-time"
    X = loaded.encode(row)
    assert X["card_network"].iloc[0] == -1


def test_missing_numeric_becomes_zero_not_nan(loaded):
    row = dict.fromkeys(ALL_MODEL_INPUTS, 1.0)
    row["dist1"] = None
    X = loaded.encode(row)
    assert X["dist1"].iloc[0] == 0.0
    assert not X.isna().any().any()


def test_row_and_frame_encoders_agree(loaded):
    """encode() serves one request; encode_frame() serves batches. They must match."""
    rows = []
    for i in range(20):
        r = {c: float(i) for c in ALL_MODEL_INPUTS}
        r.update(
            product_cd="CHRSW"[i % 5],
            card_network="visa",
            card_type="credit",
            email_domain="gmail.com",
        )
        rows.append(r)
    frame = loaded.encode_frame(pd.DataFrame(rows))
    for i, r in enumerate(rows):
        single = loaded.encode(r)
        for col in loaded.feature_order:
            assert float(single[col].iloc[0]) == pytest.approx(float(frame[col].iloc[i])), col


def test_categorical_columns_never_end_up_constant(loaded):
    """Regression guard for findings.md #11.

    A string column coerced with pd.to_numeric collapses to a single value. The
    encoder must produce variation when the inputs vary.
    """
    rows = [
        {**dict.fromkeys(ALL_MODEL_INPUTS, 1.0), "product_cd": pc}
        for pc in ("W", "C", "R", "H", "S")
    ]
    frame = loaded.encode_frame(pd.DataFrame(rows))
    for col in CATEGORICAL_FEATURE_NAMES:
        if col == "product_cd":
            assert frame[col].nunique() == 5


def test_score_request_defaults_are_serving_safe():
    req = ScoreRequest(transaction_id=1, card_id="card_1", amount=10.0)
    assert req.product_cd == "W"
    assert req.explain is False
    # 31 floats on every response is pure overhead; opt-in by design.
    assert req.include_features is False


def test_transaction_event_rejects_unknown_product_code():
    ev = TransactionEvent(
        transaction_id=1,
        card_id="c",
        event_timestamp="2018-01-01T00:00:00",
        amount=1.0,
        product_cd="ZZZ",
    )
    assert ev.product_cd == "W"


def test_transaction_event_rejects_negative_amount():
    with pytest.raises(ValueError):
        TransactionEvent(
            transaction_id=1,
            card_id="c",
            event_timestamp="2018-01-01T00:00:00",
            amount=-1.0,
            product_cd="W",
        )
