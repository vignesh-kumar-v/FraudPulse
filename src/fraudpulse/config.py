"""Central configuration.

Every knob has a working default for the local docker stack, so nothing in the
repo requires a .env file to run. Override with FP_* environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FP_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- paths -------------------------------------------------------------
    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    landing_dir: Path = REPO_ROOT / "data" / "landing"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    reports_dir: Path = REPO_ROOT / "reports"
    feature_repo_dir: Path = REPO_ROOT / "feature_repo"

    # --- kafka -------------------------------------------------------------
    kafka_bootstrap: str = "localhost:19092"
    kafka_topic: str = "transactions"
    kafka_group_landing: str = "fp-landing"
    kafka_group_features: str = "fp-features"
    kafka_partitions: int = 6

    # --- redis (feast online store) ----------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # --- mlflow ------------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment: str = "fraudpulse"
    registered_model_name: str = "fraudpulse-fraud-classifier"

    # --- serving -----------------------------------------------------------
    model_stage: str = "Production"
    score_threshold: float = 0.5
    enable_shap: bool = True

    # --- monitoring --------------------------------------------------------
    drift_window: int = 5_000
    drift_share_threshold: float = 0.3
    slack_webhook_url: str = Field(default="")

    @property
    def redis_connection_string(self) -> str:
        return f"{self.redis_host}:{self.redis_port},db={self.redis_db}"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.raw_dir,
            self.landing_dir,
            self.processed_dir,
            self.reports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
