"""Wire format for a transaction event.

One schema, used by the Kafka producer, the landing consumer, the feature
consumer and the FastAPI request body. If the shape ever drifts between them
that is a skew bug waiting to happen, so there is exactly one definition.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudpulse.features.spec import PRODUCT_CODES


class TransactionEvent(BaseModel):
    """A single card transaction as it appears on the ``transactions`` topic."""

    model_config = ConfigDict(extra="ignore")

    transaction_id: int = Field(description="IEEE-CIS TransactionID; unique per event.")
    card_id: str = Field(description="Entity key. Derived from IEEE-CIS card1.")
    event_timestamp: datetime = Field(description="Wall-clock time of the transaction (UTC).")
    amount: float = Field(ge=0, description="TransactionAmt, in USD.")
    product_cd: str = Field(description="Merchant/product category code.")
    card_network: str = Field(default="unknown", description="visa / mastercard / ...")
    card_type: str = Field(default="unknown", description="credit / debit")
    email_domain: str = Field(default="unknown")
    addr1: float | None = Field(default=None)
    dist1: float | None = Field(default=None)
    # Label. Present in the replayed historical stream so we can score the
    # model online; a production stream would receive this later, out of band.
    is_fraud: int | None = Field(default=None, ge=0, le=1)

    @field_validator("product_cd")
    @classmethod
    def _known_product(cls, v: str) -> str:
        return v if v in PRODUCT_CODES else "W"

    @property
    def epoch_seconds(self) -> float:
        return self.event_timestamp.timestamp()


class ScoreRequest(BaseModel):
    """Inference request. Same fields as the event minus the label."""

    model_config = ConfigDict(extra="ignore")

    transaction_id: int
    card_id: str
    event_timestamp: datetime | None = None
    amount: float = Field(ge=0)
    product_cd: str = "W"
    card_network: str = "unknown"
    card_type: str = "unknown"
    email_domain: str = "unknown"
    addr1: float | None = None
    dist1: float | None = None
    explain: bool = Field(default=False, description="Attach top-N SHAP contributions.")
    top_k: int = Field(default=5, ge=1, le=20)


class ShapContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class ScoreResponse(BaseModel):
    transaction_id: int
    card_id: str
    fraud_probability: float
    is_fraud_pred: bool
    threshold: float
    model_version: str
    feature_source: str = Field(description="'online_store' or 'cold_start'")
    latency_ms: float
    features_used: dict[str, float] | None = None
    explanation: list[ShapContribution] | None = None
