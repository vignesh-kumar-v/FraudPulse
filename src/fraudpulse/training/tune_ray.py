"""Stretch (Phase 5) — the same search, run in parallel with Ray Tune.

The point of this module is a measurement, not a framework swap: how much
wall-clock does distributing the search actually buy on one machine, once you
pay for Ray's scheduling and for the workers contending over the same cores?

Two details that decide whether the comparison is honest:

* **The search space and sampler are the same as Optuna's.** Ray Tune drives
  ``OptunaSearch`` over the identical distributions, so the only variable is
  parallelism. Swapping the sampler as well would make the speedup
  uninterpretable.
* **Per-trial threads are capped.** XGBoost defaults to every core. Run four
  trials at once and you get 4x oversubscription, each trial slowing the others
  down, and a "speedup" that is mostly thread thrash. ``n_jobs`` is set to
  ``cores // concurrency`` so the comparison measures parallelism rather than
  contention.
"""

from __future__ import annotations

import os
import time

from sklearn.metrics import average_precision_score

from fraudpulse.logging_utils import get_logger
from fraudpulse.training.dataset import Split

log = get_logger(__name__)


def _space(model_type: str) -> dict:
    from ray import tune

    if model_type == "xgboost":
        return {
            "n_estimators": 1500,
            "max_depth": tune.randint(3, 11),
            "learning_rate": tune.loguniform(0.01, 0.3),
            "subsample": tune.uniform(0.5, 1.0),
            "colsample_bytree": tune.uniform(0.5, 1.0),
            "min_child_weight": tune.loguniform(0.5, 20.0),
            "reg_lambda": tune.loguniform(1e-3, 30.0),
            "reg_alpha": tune.loguniform(1e-4, 10.0),
        }
    return {
        "n_estimators": 2000,
        "num_leaves": tune.randint(15, 256),
        "learning_rate": tune.loguniform(0.01, 0.3),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        "min_child_samples": tune.randint(5, 200),
        "reg_lambda": tune.loguniform(1e-3, 30.0),
        "reg_alpha": tune.loguniform(1e-4, 10.0),
    }


def tune_ray(
    model_type: str,
    split: Split,
    n_trials: int,
    *,
    concurrency: int | None = None,
) -> tuple[dict, float, float]:
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    cores = os.cpu_count() or 4
    concurrency = concurrency or max(2, min(6, cores // 2))
    threads_per_trial = max(1, cores // concurrency)
    log.info(
        "ray tune: %d trials, concurrency=%d, %d threads/trial (%d cores)",
        n_trials,
        concurrency,
        threads_per_trial,
        cores,
    )

    if not ray.is_initialized():
        ray.init(
            num_cpus=cores, log_to_driver=False, include_dashboard=False, ignore_reinit_error=True
        )

    # Put the split in the object store once instead of pickling it into every
    # trial. Without this, each trial ships ~100MB of DataFrame and the
    # "parallel" run spends its time in serialisation.
    handle = ray.put(split)

    def objective(config: dict) -> None:
        from fraudpulse.training.train import fit_model

        s: Split = ray.get(handle)
        params = {
            k: (
                int(v)
                if k in {"max_depth", "num_leaves", "min_child_samples", "n_estimators"}
                else v
            )
            for k, v in config.items()
        }
        params["n_jobs"] = threads_per_trial
        model = fit_model(model_type, params, s)
        score = average_precision_score(s.y_valid, model.predict_proba(s.X_valid)[:, 1])
        tune.report({"pr_auc": float(score)})

    t0 = time.perf_counter()
    tuner = tune.Tuner(
        tune.with_resources(objective, {"cpu": threads_per_trial}),
        param_space=_space(model_type),
        tune_config=tune.TuneConfig(
            metric="pr_auc",
            mode="max",
            num_samples=n_trials,
            search_alg=OptunaSearch(metric="pr_auc", mode="max", seed=42),
            max_concurrent_trials=concurrency,
        ),
        run_config=ray.tune.RunConfig(verbose=0),
    )
    results = tuner.fit()
    elapsed = time.perf_counter() - t0

    best = results.get_best_result(metric="pr_auc", mode="max")
    params = dict(best.config)
    score = float(best.metrics["pr_auc"])
    log.info("ray tune: %d trials in %.1fs, best valid PR-AUC=%.5f", n_trials, elapsed, score)
    return params, score, elapsed


def compare_sequential_vs_parallel(model_type: str = "xgboost", n_trials: int = 20) -> dict:
    """Phase 5 verification: measured wall-clock, both ways, same search space.

    Reports the real number including Ray's startup cost, because that cost is
    real for anyone running this.
    """
    import json

    from fraudpulse.config import settings
    from fraudpulse.training.dataset import chronological_split, load_or_build
    from fraudpulse.training.train import tune_optuna

    split = chronological_split(load_or_build())

    _, seq_score, seq_s = tune_optuna(model_type, split, n_trials)
    _, par_score, par_s = tune_ray(model_type, split, n_trials)

    result = {
        "model_type": model_type,
        "n_trials": n_trials,
        "cores": os.cpu_count(),
        "sequential_optuna_seconds": seq_s,
        "parallel_ray_seconds": par_s,
        "speedup": seq_s / par_s if par_s else float("nan"),
        "sequential_best_pr_auc": seq_score,
        "parallel_best_pr_auc": par_score,
        "pr_auc_delta": par_score - seq_score,
    }
    out = settings.reports_dir / "hpo_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(
        "HPO %d trials: optuna %.1fs (PR-AUC %.5f) vs ray %.1fs (PR-AUC %.5f) -> %.2fx wall-clock",
        n_trials,
        seq_s,
        seq_score,
        par_s,
        par_score,
        result["speedup"],
    )
    return result


__all__ = ["compare_sequential_vs_parallel", "tune_ray"]
