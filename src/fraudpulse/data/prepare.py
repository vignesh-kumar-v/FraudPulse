"""Turn the raw IEEE-CIS CSV into the canonical FraudPulse event table.

The canonical table is the contract for everything downstream: the Kafka
producer replays it, the offline feature builder reads it, and the parity check
compares against it. Doing the messy dataset-specific work exactly once, here,
keeps IEEE-CIS quirks out of the rest of the codebase.

IEEE-CIS specifics handled here:
  * ``TransactionDT`` is an integer offset in seconds from an undisclosed epoch.
    Anchored to 2017-12-01 (the community-accepted reference) so the data has a
    real wall clock for windowing, plots and Feast TTLs.
  * ``card1`` is the card/account proxy and becomes our ``card_id`` entity.
  * ``ProductCD`` is the merchant-category proxy; it has exactly 5 values.
  * Roughly 3.5% of the file is fraud (``isFraud``), which is why every metric
    in this project is PR-AUC and not accuracy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudpulse.config import settings
from fraudpulse.features.spec import DATASET_EPOCH, PRODUCT_CODES
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)

RAW_CSV = "train_transaction.csv"
EVENTS_PARQUET = "events.parquet"

USECOLS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card4",
    "card6",
    "addr1",
    "dist1",
    "P_emaildomain",
    "isFraud",
]

CANONICAL_COLUMNS = [
    "transaction_id",
    "card_id",
    "event_timestamp",
    "amount",
    "product_cd",
    "card_network",
    "card_type",
    "email_domain",
    "addr1",
    "dist1",
    "is_fraud",
]


def prepare_events(
    raw_csv: Path | None = None,
    out_parquet: Path | None = None,
    *,
    sample_cards: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Read the raw CSV and write ``data/processed/events.parquet``.

    ``sample_cards`` takes a random subset of *cards* (not rows). Sampling rows
    would silently shred every card's transaction history and make the rolling
    windows meaningless - the kind of shortcut that produces a feature pipeline
    that "works" on the sample and collapses on the full data.
    """
    raw_csv = raw_csv or settings.raw_dir / RAW_CSV
    out_parquet = out_parquet or settings.processed_dir / EVENTS_PARQUET
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run `make data` (kaggle competitions download "
            "-c ieee-fraud-detection -f train_transaction.csv)."
        )

    log.info("reading %s", raw_csv)
    df = pd.read_csv(raw_csv, usecols=USECOLS)
    log.info("raw rows=%d cols=%d", len(df), df.shape[1])

    epoch = pd.Timestamp(DATASET_EPOCH)
    out = pd.DataFrame(
        {
            "transaction_id": df["TransactionID"].astype("int64"),
            "card_id": "card_" + df["card1"].fillna(-1).astype("int64").astype(str),
            "event_timestamp": epoch + pd.to_timedelta(df["TransactionDT"], unit="s"),
            "amount": df["TransactionAmt"].astype("float64"),
            "product_cd": df["ProductCD"]
            .astype(str)
            .where(df["ProductCD"].isin(PRODUCT_CODES), "W"),
            "card_network": df["card4"].fillna("unknown").astype(str),
            "card_type": df["card6"].fillna("unknown").astype(str),
            "email_domain": df["P_emaildomain"].fillna("unknown").astype(str),
            "addr1": df["addr1"].astype("float64"),
            "dist1": df["dist1"].astype("float64"),
            "is_fraud": df["isFraud"].astype("int8"),
        }
    )

    if sample_cards is not None:
        cards = (
            out["card_id"]
            .drop_duplicates()
            .sample(n=min(sample_cards, out["card_id"].nunique()), random_state=seed)
        )
        before = len(out)
        out = out[out["card_id"].isin(set(cards))].copy()
        log.info("sampled %d cards: %d -> %d rows", len(cards), before, len(out))

    # Global time order is what makes the replay a believable stream.
    out = out.sort_values(["event_timestamp", "transaction_id"], kind="stable").reset_index(
        drop=True
    )

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    fraud_rate = out["is_fraud"].mean()
    log.info(
        "wrote %s  rows=%d  cards=%d  span=%s..%s  fraud_rate=%.4f",
        out_parquet,
        len(out),
        out["card_id"].nunique(),
        out["event_timestamp"].min(),
        out["event_timestamp"].max(),
        fraud_rate,
    )
    return out


def load_events(path: Path | None = None) -> pd.DataFrame:
    path = path or settings.processed_dir / EVENTS_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make prepare` first.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    import typer

    typer.run(lambda sample_cards: prepare_events(sample_cards=sample_cards or None))
