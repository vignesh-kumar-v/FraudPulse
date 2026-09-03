#!/usr/bin/env python
"""Training entrypoint for the Fargate task.

Reads the pre-joined training Parquet from S3, fits XGBoost on the same
chronological split the local run uses, and writes the model plus a metrics
JSON back to S3. Credentials come from the ECS task role - there is no key
material anywhere in the image or the task definition.

The split boundaries and the metric are duplicated from
``fraudpulse.training`` on purpose: this container deliberately does not depend
on the project package, so that the cloud path stays a thin, auditable script
rather than dragging Feast, Kafka and MLflow into a batch image. The tradeoff is
that these two constants have to agree, so the job re-reports PR-AUC and
``scripts/run_fargate_training.py`` compares it to the local number.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

BUCKET = os.environ["FP_S3_BUCKET"]
TRAINING_KEY = os.environ.get("FP_TRAINING_KEY", "training/training_set.parquet")
OUTPUT_PREFIX = os.environ.get("FP_OUTPUT_PREFIX", "training/output")
LABEL = "is_fraud"
VALID_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_STATE = 42
ID_COLUMNS = {LABEL, "transaction_id", "card_id", "event_timestamp"}
# Must match fraudpulse.features.spec.CATEGORICAL_FEATURE_NAMES.
CATEGORICAL_COLUMNS = ("product_cd", "card_network", "card_type", "email_domain")


def log(msg: str) -> None:
    print(f"[trainer] {msg}", flush=True)


def _encode(df: pd.DataFrame, feature_cols: list[str], *, train_end: int) -> pd.DataFrame:
    """Numeric matrix, with categoricals ordinal-encoded from the TRAIN slice.

    The first version of this file was one line - ``pd.to_numeric(errors=
    "coerce").fillna(0.0)`` across every feature column. On the four string
    columns that silently produced a constant 0.0, so the cloud model trained on
    27 real features and 4 dead ones, including `product_cd`, the single
    highest-importance feature in the local model. It cost 0.029 PR-AUC and
    raised no error of any kind. See docs/findings.md #11.
    """
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in CATEGORICAL_COLUMNS:
            cats = sorted(df[col].astype(str).iloc[:train_end].unique())
            lookup = {v: i for i, v in enumerate(cats)}
            out[col] = df[col].astype(str).map(lookup).fillna(-1).astype("int32")
            log(f"encoded {col}: {len(cats)} categories from the train slice")
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
    return out[feature_cols]


def _assert_no_dead_features(X: pd.DataFrame) -> None:
    """Refuse to train on a column that carries no information.

    A constant feature is almost never intentional; it is what a botched
    encoding, a failed join or a missing column looks like from the inside.
    Cheap to check, and it is the guard that would have caught #11 on the first
    run instead of on the PR-AUC comparison afterwards.
    """
    dead = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if dead:
        raise SystemExit(
            f"[trainer] ABORT: {len(dead)} feature(s) are constant across the "
            f"training slice and carry no signal: {dead}. This is a data or "
            "encoding bug, not a model that happens to be bad."
        )


def main() -> int:
    s3 = boto3.client("s3")

    log(f"reading s3://{BUCKET}/{TRAINING_KEY}")
    obj = s3.get_object(Bucket=BUCKET, Key=TRAINING_KEY)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    log(f"{len(df)} rows, {df.shape[1]} columns")

    df = df.sort_values("event_timestamp", kind="stable").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ID_COLUMNS]
    y = df[LABEL].astype(int)

    n = len(df)
    i_valid = int(n * (1 - VALID_FRAC - TEST_FRAC))
    i_test = int(n * (1 - TEST_FRAC))
    log(f"chronological split: train={i_valid} valid={i_test - i_valid} test={n - i_test}")

    X = _encode(df, feature_cols, train_end=i_valid)
    _assert_no_dead_features(X.iloc[:i_valid])

    pos = float(y.iloc[:i_valid].sum())
    spw = (i_valid - pos) / max(pos, 1.0)

    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    t0 = time.perf_counter()
    model.fit(
        X.iloc[:i_valid],
        y.iloc[:i_valid],
        eval_set=[(X.iloc[i_valid:i_test], y.iloc[i_valid:i_test])],
        verbose=False,
    )
    fit_s = time.perf_counter() - t0
    log(f"fit in {fit_s:.1f}s, best_iteration={model.best_iteration}")

    scores = model.predict_proba(X.iloc[i_test:])[:, 1]
    y_test = y.iloc[i_test:].to_numpy()
    metrics = {
        "test_pr_auc": float(average_precision_score(y_test, scores)),
        "test_roc_auc": float(roc_auc_score(y_test, scores)),
        "baseline_prevalence": float(y_test.mean()),
        "baseline_amount_only_pr_auc": float(
            average_precision_score(y_test, X["amount"].iloc[i_test:].to_numpy())
        ),
        "best_iteration": int(model.best_iteration),
        "fit_seconds": round(fit_s, 2),
        "n_train": int(i_valid),
        "n_test": int(n - i_test),
        "n_features": len(feature_cols),
        "feature_order": feature_cols,
        "xgboost_version": xgb.__version__,
        "numpy_version": np.__version__,
        "categorical_columns": list(CATEGORICAL_COLUMNS),
    }
    log(json.dumps({k: v for k, v in metrics.items() if k != "feature_order"}))

    model_path = "/tmp/model.json"
    model.save_model(model_path)
    with open(model_path, "rb") as fh:
        s3.put_object(Bucket=BUCKET, Key=f"{OUTPUT_PREFIX}/model.json", Body=fh.read())
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}/metrics.json",
        Body=json.dumps(metrics, indent=2).encode(),
        ContentType="application/json",
    )
    log(f"wrote s3://{BUCKET}/{OUTPUT_PREFIX}/model.json and metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
