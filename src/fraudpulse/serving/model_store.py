"""Load the registered model and everything needed to reproduce training's view.

A model artifact alone is not enough to serve without skew. The three things
that also have to travel with it:

  * ``feature_order`` — the exact column order the model was fitted on. Trees do
    not name their inputs; hand them the same values in a different order and
    you get confident, silently wrong predictions.
  * ``categories``   — the string -> integer map used at training time.
    Re-deriving it from live traffic is how ``visa`` becomes 2 in production and
    0 in training.
  * ``model_type``   — which flavour to load back.

All three are written as ``serving_metadata.json`` next to the model in MLflow,
so a deployment cannot pick up a model without them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraudpulse.config import settings
from fraudpulse.features.spec import ALL_MODEL_INPUTS, CATEGORICAL_FEATURE_NAMES
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class LoadedModel:
    model: Any
    version: str
    feature_order: list[str]
    categories: dict[str, list[str]]
    model_type: str
    threshold: float
    test_pr_auc: float | None = None

    def encode(self, row: dict[str, float | str]) -> pd.DataFrame:
        """Turn a raw feature dict into the exact frame the model was fitted on."""
        out: dict[str, Any] = {}
        for col in self.feature_order:
            v = row.get(col)
            if col in CATEGORICAL_FEATURE_NAMES:
                lookup = self._lookup(col)
                out[col] = np.int32(lookup.get(str(v), -1))
            else:
                out[col] = np.float64(0.0 if v is None else v)
        return pd.DataFrame([out], columns=self.feature_order)

    def encode_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.feature_order:
            if col in CATEGORICAL_FEATURE_NAMES:
                lookup = self._lookup(col)
                out[col] = df[col].astype(str).map(lookup).fillna(-1).astype("int32")
            else:
                out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
        return out[self.feature_order]

    def _lookup(self, col: str) -> dict[str, int]:
        return {v: i for i, v in enumerate(self.categories.get(col, []))}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def load_latest(model_name: str | None = None) -> LoadedModel:
    """Load the newest version of the registered model plus its serving metadata."""
    import mlflow
    from mlflow.tracking import MlflowClient

    name = model_name or settings.registered_model_name
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(
            f"no versions of model '{name}' in the registry at "
            f"{settings.mlflow_tracking_uri}. Run `make train` first."
        )
    latest = max(versions, key=lambda v: int(v.version))
    log.info("loading %s v%s (run %s)", name, latest.version, latest.run_id)

    meta = _load_metadata(client, latest.run_id)
    model_type = meta.get("model_type", "xgboost")
    flavour = mlflow.xgboost if model_type == "xgboost" else mlflow.lightgbm
    model = flavour.load_model(f"models:/{name}/{latest.version}")

    order = meta.get("feature_order") or ALL_MODEL_INPUTS
    if order != ALL_MODEL_INPUTS:
        # Not fatal - an older model can legitimately have a different feature
        # set - but it must be loud, because it is exactly the situation where a
        # code change has moved ahead of the deployed model.
        log.warning(
            "model feature_order differs from the current spec "
            "(model has %d columns, spec has %d). Serving the model's order.",
            len(order), len(ALL_MODEL_INPUTS),
        )

    return LoadedModel(
        model=model,
        version=str(latest.version),
        feature_order=order,
        categories=meta.get("categories", {}),
        model_type=model_type,
        threshold=float(meta.get("threshold", settings.score_threshold)),
        test_pr_auc=meta.get("test_pr_auc"),
    )


def _load_metadata(client, run_id: str) -> dict:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = client.download_artifacts(run_id, "serving_metadata.json", tmp)
        except Exception as exc:
            log.warning("no serving_metadata.json on run %s (%s); falling back to the "
                        "current spec, which may not match the model", run_id, exc)
            return {}
        return json.loads(Path(path).read_text())
