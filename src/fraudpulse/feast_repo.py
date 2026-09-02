"""Thin wrapper around the Feast repo: apply, write the offline parquet, read back.

Keeping this in one module means nothing else in the codebase has to know where
the registry lives or what the push source is called.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from fraudpulse.config import settings
from fraudpulse.features.spec import ENTITY_KEY, EVENT_TS, FEATURE_DTYPES, FEATURE_NAMES
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)

OFFLINE_PARQUET = "card_features.parquet"
PUSH_SOURCE = "card_stats_push"
FEATURE_SERVICE = "fraud_serving_v1"


def repo_path() -> Path:
    return settings.feature_repo_dir


def write_offline_parquet(features: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist the offline feature values in the layout the FileSource expects.

    ``created`` is required by the FileSource's ``created_timestamp_column``;
    Feast uses it to break ties when two rows share an event timestamp for the
    same entity, which - given this dataset has plenty of those - is not
    hypothetical.
    """
    path = path or repo_path() / "data" / OFFLINE_PARQUET
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [ENTITY_KEY, EVENT_TS, *FEATURE_NAMES]
    out = features[cols].copy()
    out[EVENT_TS] = pd.to_datetime(out[EVENT_TS])
    out["created"] = pd.Timestamp.utcnow().tz_localize(None)
    for name, dtype in FEATURE_DTYPES.items():
        out[name] = out[name].astype(dtype)
    out.to_parquet(path, index=False)
    log.info("wrote offline feature parquet: %s (%d rows)", path, len(out))
    return path


def apply_repo() -> None:
    """``feast apply`` — register entities/views/services into the registry."""
    log.info("running `feast apply` in %s", repo_path())
    proc = subprocess.run(
        [str(settings.repo_root / ".venv" / "bin" / "feast"), "apply"],
        cwd=repo_path(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        log.error("feast apply failed:\n%s\n%s", proc.stdout, proc.stderr)
        raise RuntimeError("feast apply failed")
    log.info("feast apply ok\n%s", proc.stdout.strip())


def get_store():
    from feast import FeatureStore

    return FeatureStore(repo_path=str(repo_path()))
