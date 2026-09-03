"""Phase 4 — feature drift monitoring with Evidently.

What is compared, and why it is not the obvious thing:

The reference is the **training feature distribution** — the exact frame the
model was fitted on. The current window is the **features the online path
served**, read from the scoring-view capture the Kafka consumer writes. So this
watches the quantity that actually matters: has the input the *model sees in
production* moved away from the input it was *fitted on*. Comparing raw incoming
transactions against raw training transactions would miss drift introduced by
the feature pipeline itself, which is a large part of what can go wrong.

Verification is adversarial by design (`run_drift_experiment`): the monitor is
first run on an unshifted window, where it must stay quiet, and then on windows
with three different injected shifts, where it must fire. A monitor that has
only ever been shown drifting data has not been tested — it has been assumed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fraudpulse.config import settings
from fraudpulse.features.spec import MODEL_STORE_FEATURE_NAMES, ONDEMAND_FEATURE_NAMES
from fraudpulse.logging_utils import get_logger
from fraudpulse.monitoring.alert import fire

log = get_logger(__name__)

MONITORED_COLUMNS = [*MODEL_STORE_FEATURE_NAMES, *ONDEMAND_FEATURE_NAMES, "amount"]

# Features whose drift alerts on its own, regardless of the overall share.
# These are the ones the model leans on hardest and the ones a fraud campaign
# moves first.
KEY_FEATURES = [
    "amount",
    "amt_mean_24h",
    "amt_over_mean_24h",
    "txn_count_1h",
    "seconds_since_last_txn",
    "product_freq_7d",
]


@dataclass
class DriftResult:
    scenario: str
    n_reference: int
    n_current: int
    n_columns: int
    n_drifted: int
    drift_share: float
    threshold: float
    alerted: bool
    key_features_drifted: list[str] = field(default_factory=list)
    drifted_columns: list[str] = field(default_factory=list)
    per_column: dict[str, float] = field(default_factory=dict)
    report_html: str | None = None

    def summary(self) -> str:
        head = (
            f"[{self.scenario}] {self.n_drifted}/{self.n_columns} columns drifted "
            f"(share={self.drift_share:.3f}, threshold={self.threshold:.2f}) "
            f"-> {'ALERT' if self.alerted else 'quiet'}"
        )
        if self.key_features_drifted:
            head += f"\n    key features drifted: {', '.join(self.key_features_drifted)}"
        if not self.drifted_columns:
            return head
        worst = sorted(
            ((c, self.per_column.get(c, 0.0)) for c in self.drifted_columns),
            key=lambda kv: -abs(kv[1]),
        )[:6]
        return head + "\n" + "\n".join(
            f"    {c:<26} drift_score={v:.4g}" for c, v in worst
        )


def _to_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in MONITORED_COLUMNS if c in df.columns]
    return df[cols].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    scenario: str = "live",
    threshold: float | None = None,
    html_path: Path | None = None,
) -> DriftResult:
    """Run Evidently's data-drift preset and reduce it to one alerting decision."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    threshold = settings.drift_share_threshold if threshold is None else threshold
    ref, cur = _to_frame(reference), _to_frame(current)
    common = [c for c in ref.columns if c in cur.columns]
    ref, cur = ref[common], cur[common]

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=cur, reference_data=ref)
    payload = json.loads(snapshot.json())

    per_column, drifted, evidently_count = _extract_column_drift(payload, common)
    n_drifted = int(evidently_count) if evidently_count is not None else len(drifted)
    share = n_drifted / len(common) if common else 0.0

    # Two ways to alert. A share threshold catches broad distribution shift; a
    # named watchlist catches the case a fraud team actually cares about, where
    # one load-bearing feature moves and the other twenty-five do not - which a
    # share rule alone would average away to nothing.
    key_drifted = sorted(set(drifted) & set(KEY_FEATURES))
    alerted = share >= threshold or bool(key_drifted)

    html = None
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
        html = str(html_path)

    result = DriftResult(
        scenario=scenario,
        n_reference=len(ref),
        n_current=len(cur),
        n_columns=len(common),
        n_drifted=n_drifted,
        drift_share=share,
        threshold=threshold,
        alerted=alerted,
        key_features_drifted=key_drifted,
        drifted_columns=drifted,
        per_column=per_column,
        report_html=html,
    )
    log.info(result.summary())
    if alerted:
        fire(
            f"feature drift detected ({scenario})",
            result.summary(),
            severity="critical" if share >= 2 * threshold else "warning",
        )
    return result


def _extract_column_drift(
    payload: dict, columns: list[str]
) -> tuple[dict[str, float], list[str], float | None]:
    """Pull per-column drift out of Evidently's snapshot JSON.

    Two things make this less trivial than it looks.

    **The comparison direction depends on the method.** Evidently picks a drift
    test per column based on sample size and cardinality. Statistical tests
    ("K-S p_value", "chi-square p_value") report a p-value and drift means
    *below* the threshold; distance measures ("Wasserstein distance (normed)",
    "Jensen-Shannon distance", "PSI") report a magnitude and drift means *above*
    it. Hardcoding ``< 0.05`` - the obvious thing - silently inverts the verdict
    on every column Evidently decided to measure with a distance, which on
    frames this size is all of them.

    **Key names move between releases.** So the count Evidently computed itself
    (``DriftedColumnsCount``) is read as well, and disagreement with our own
    tally is logged loudly rather than papered over. A drift monitor that
    reports "no drift" because a JSON key was renamed is worse than no monitor.
    """
    per_column: dict[str, float] = {}
    drifted: list[str] = []
    evidently_count: float | None = None

    for m in payload.get("metrics", []):
        name = str(m.get("metric_name", ""))
        cfg = m.get("config", {}) or {}
        value = m.get("value")

        if name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                evidently_count = float(value.get("count", 0.0))
            continue

        if not name.startswith("ValueDrift"):
            continue
        col = cfg.get("column")
        if col not in columns or not isinstance(value, (int, float)):
            continue

        per_column[col] = float(value)
        threshold = float(cfg.get("threshold", 0.05))
        method = str(cfg.get("method", "")).lower()
        is_pvalue = "p_value" in method or "test" in method and "distance" not in method
        if (value < threshold) if is_pvalue else (value > threshold):
            drifted.append(col)

    if not per_column:
        log.error(
            "no ValueDrift metrics found in the Evidently snapshot - the JSON layout "
            "has changed and this monitor is blind. Refusing to report 'no drift'."
        )
        raise RuntimeError("could not read per-column drift from the Evidently snapshot")

    if evidently_count is not None and int(evidently_count) != len(drifted):
        log.warning(
            "drift tally disagrees with Evidently's own DriftedColumnsCount "
            "(ours=%d, theirs=%d) - trusting Evidently",
            len(drifted), int(evidently_count),
        )
    return per_column, sorted(set(drifted)), evidently_count


# --------------------------------------------------------------------------
# Phase 4 verification
# --------------------------------------------------------------------------
def inject_shift(events: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Apply a controlled distribution shift to real transactions.

    Injecting into the *real* stream, rather than swapping in synthetic data, is
    what makes the null control meaningful: the unshifted window and every
    shifted window come from the same source, so the only thing that differs is
    the injection. Comparing real training data against a synthetic window would
    fire on everything and prove nothing.
    """
    out = events.copy()
    if kind == "none":
        return out
    if kind == "amount":
        # A campaign moving to higher-value transactions.
        out["amount"] = out["amount"] * 4.0
    elif kind == "velocity":
        # Card-testing: the same cards transacting ~20x faster. Rebuild each
        # card's inter-arrival gaps rather than shifting timestamps globally,
        # so the change lands in the rolling windows and not in the calendar.
        out = out.sort_values(["card_id", "event_timestamp"], kind="stable")
        gaps = out.groupby("card_id")["event_timestamp"].diff().fillna(pd.Timedelta(0))
        compressed = (gaps / 20).groupby(out["card_id"]).cumsum()
        starts = out.groupby("card_id")["event_timestamp"].transform("min")
        out["event_timestamp"] = starts + compressed
        out = out.sort_values(["event_timestamp", "transaction_id"], kind="stable")
    elif kind == "product":
        # Merchant-category mix collapsing onto one code.
        out["product_cd"] = "C"
    else:
        raise ValueError(f"unknown shift {kind!r}")
    return out.reset_index(drop=True)


def _feature_window(events: pd.DataFrame, kind: str, tail_frac: float) -> pd.DataFrame:
    """Features for the last ``tail_frac`` of the stream, under shift ``kind``.

    Features are computed over the *whole* shifted stream and only then sliced,
    so each event still sees its own card's real history. Injecting into the
    tail alone would give every card an artificial cold start and manufacture
    drift that has nothing to do with the shift being tested.
    """
    from fraudpulse.features.offline import compute_offline_features
    from fraudpulse.features.ondemand import compute_ondemand_frame

    shifted = inject_shift(events, kind)
    feats = compute_offline_features(shifted)
    feats = pd.concat([feats, compute_ondemand_frame(feats)], axis=1)
    n_tail = int(len(feats) * tail_frac)
    return feats.tail(n_tail).reset_index(drop=True)


def run_drift_experiment(
    *, inject: str = "all", out_dir: Path | None = None, tail_frac: float = 0.15
) -> dict:
    """Prove the monitor fires on injected shift, and separate that from the
    drift the data does on its own.

    Three kinds of comparison, because the first attempt at this collapsed two
    of them and produced a misleading answer:

    ``null`` (must stay quiet)
        Two disjoint *random* halves of the training slice. Same period, same
        process, so any alert here is the monitor over-firing.

    ``temporal`` (expected to fire; not a failure)
        Training slice vs. the untouched later slice. IEEE-CIS spans six months
        and genuinely drifts over them - the first version of this experiment
        used this as its "no shift" control, saw 7 of 23 columns drift, and
        looked like a false-positive rate of 30%. It was not: the monitor was
        correctly reporting real seasonality. Reported here as the *floor* an
        injected shift has to clear to be meaningful.

    ``amount`` / ``velocity`` / ``product`` (must fire, and must exceed the floor)
        The later slice with a controlled shift applied.

    Passing requires all three: the null window quiet, every injection alerting,
    and every injection drifting more than the temporal floor.
    """
    from fraudpulse.data.prepare import load_events
    from fraudpulse.features.ondemand import compute_ondemand_frame
    from fraudpulse.training.dataset import load_or_build

    out_dir = out_dir or settings.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    training = load_or_build().sort_values("event_timestamp", kind="stable")
    events = load_events()
    n_ref = int(len(training) * (1 - tail_frac))

    # Reference is the TRAINING slice only. Using the whole training set would
    # include the very window being tested, which quietly deflates the result.
    ref_slice = training.iloc[:n_ref]
    reference = pd.concat([ref_slice, compute_ondemand_frame(ref_slice)], axis=1) \
        if "amt_over_mean_24h" not in ref_slice.columns else ref_slice
    log.info("reference = first %.0f%% of training features (%d rows)",
             (1 - tail_frac) * 100, len(reference))

    results: list[DriftResult] = []

    # --- null: two random halves of the same period ------------------------
    shuffled = reference.sample(frac=1.0, random_state=7)
    half = len(shuffled) // 2
    results.append(
        detect_drift(shuffled.iloc[:half], shuffled.iloc[half:], scenario="null",
                     html_path=out_dir / "drift_null.html")
    )

    # --- temporal + injections --------------------------------------------
    scenarios = ["none"] + (["amount", "velocity", "product"] if inject == "all" else [inject])
    for sc in scenarios:
        cur = _feature_window(events, sc, tail_frac)
        name = "temporal" if sc == "none" else sc
        results.append(
            detect_drift(reference, cur, scenario=name, html_path=out_dir / f"drift_{name}.html")
        )

    by = {r.scenario: r for r in results}
    null, temporal = by["null"], by["temporal"]
    shifted = [r for k, r in by.items() if k not in {"null", "temporal"}]

    passed = (
        not null.alerted
        and all(r.alerted for r in shifted)
        and all(r.drift_share > temporal.drift_share for r in shifted)
    )
    verdict = {
        "passed": passed,
        "null_alerted": null.alerted,
        "null_drift_share": null.drift_share,
        "temporal_floor_drift_share": temporal.drift_share,
        "temporal_alerted": temporal.alerted,
        "detected": {r.scenario: r.alerted for r in shifted},
        "clears_temporal_floor": {
            r.scenario: r.drift_share > temporal.drift_share for r in shifted
        },
        "true_positive_rate": (sum(r.alerted for r in shifted) / len(shifted)) if shifted else 0.0,
        "results": [asdict(r) for r in results],
    }

    out = out_dir / "drift_experiment.json"
    out.write_text(json.dumps(verdict, indent=2, default=str))
    log.info("drift experiment %s -> %s", "PASSED" if passed else "FAILED", out)
    for r in results:
        log.info("  %-10s share=%.3f alert=%s key=%s",
                 r.scenario, r.drift_share, r.alerted, r.key_features_drifted or "-")
    return verdict


def detect_on_served_window(*, window: int | None = None) -> DriftResult:
    """Compare the last N feature vectors the online path served to the training set.

    This is the production-shaped call: reference = what the model was fitted
    on, current = what the store actually handed the model.
    """
    from fraudpulse.features.ondemand import compute_ondemand_frame
    from fraudpulse.training.dataset import load_or_build

    window = window or settings.drift_window
    served_path = settings.processed_dir / "online_scoring_view.parquet"
    if not served_path.exists():
        raise FileNotFoundError(f"{served_path} missing; run `make features` first.")

    reference = load_or_build()
    served = pd.read_parquet(served_path).tail(window)
    events = pd.read_parquet(settings.processed_dir / "events.parquet",
                             columns=["transaction_id", "amount", "product_cd"])
    served = served.merge(events, on="transaction_id", how="left")
    served = pd.concat([served, compute_ondemand_frame(served)], axis=1)

    return detect_drift(
        reference, served, scenario="served",
        html_path=settings.reports_dir / "drift_served.html",
    )
