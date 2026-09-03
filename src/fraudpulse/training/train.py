"""Train, tune and register the fraud classifier.

Metric choice is not a detail here. The dataset is 3.5% fraud, so accuracy is
meaningless (predict "legit" every time: 96.5% accurate, 0 fraud caught) and
ROC-AUC is misleadingly flattering on heavy imbalance because the enormous
negative class makes the false-positive rate move slowly. **PR-AUC (average
precision)** is the headline number, reported against two baselines:

  * the prevalence floor (what a random ranker scores), and
  * amount-only, a one-feature model, which is the honest "did the feature store
    actually buy anything?" comparison.

Tuning optimises validation PR-AUC. The test split is chronologically last and
is scored exactly once, at the end.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from fraudpulse.config import settings
from fraudpulse.features.spec import ALL_MODEL_INPUTS
from fraudpulse.logging_utils import get_logger
from fraudpulse.training.dataset import Split, chronological_split, load_or_build

log = get_logger(__name__)

RANDOM_STATE = 42


@dataclass
class ModelResult:
    model_type: str
    params: dict[str, Any]
    valid_pr_auc: float
    test_pr_auc: float
    test_roc_auc: float
    test_precision_at_recall_50: float
    test_recall_at_precision_90: float
    fit_seconds: float
    n_trials: int
    tuning_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    # Precision achievable while catching half the fraud, and recall achievable
    # at 90% precision. Both are what a fraud team actually negotiates over;
    # a single AUC number hides the shape of that tradeoff.
    p_at_r50 = float(prec[rec >= 0.5].max()) if (rec >= 0.5).any() else 0.0
    r_at_p90 = float(rec[prec >= 0.9].max()) if (prec >= 0.9).any() else 0.0
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision_at_recall_50": p_at_r50,
        "recall_at_precision_90": r_at_p90,
    }


def baselines(split: Split) -> dict[str, float]:
    """Two floors every reported number has to clear to mean anything."""
    y = split.y_test.to_numpy()
    prevalence = float(y.mean())

    rng = np.random.default_rng(RANDOM_STATE)
    random_ap = float(average_precision_score(y, rng.random(len(y))))

    # amount alone: the cheapest possible "feature", and a surprisingly strong
    # one on fraud data. If the full model does not clearly beat this, the
    # feature store bought nothing.
    amount_ap = float(average_precision_score(y, split.X_test["amount"].to_numpy()))

    return {
        "baseline_prevalence": prevalence,
        "baseline_random_pr_auc": random_ap,
        "baseline_amount_only_pr_auc": amount_ap,
    }


# --------------------------------------------------------------------------
# model factories
# --------------------------------------------------------------------------
def _scale_pos_weight(y: pd.Series) -> float:
    pos = float(y.sum())
    return (len(y) - pos) / max(pos, 1.0)


def make_xgb(params: dict, y_train: pd.Series):
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 600),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        min_child_weight=params.get("min_child_weight", 1.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        scale_pos_weight=_scale_pos_weight(y_train),
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_lgbm(params: dict, y_train: pd.Series):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 800),
        num_leaves=params.get("num_leaves", 63),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.9),
        subsample_freq=1,
        colsample_bytree=params.get("colsample_bytree", 0.9),
        min_child_samples=params.get("min_child_samples", 20),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        scale_pos_weight=_scale_pos_weight(y_train),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def fit_model(model_type: str, params: dict, split: Split):
    if model_type == "xgboost":
        m = make_xgb(params, split.y_train)
        m.fit(split.X_train, split.y_train,
              eval_set=[(split.X_valid, split.y_valid)], verbose=False)
    elif model_type == "lightgbm":
        import lightgbm as lgb

        m = make_lgbm(params, split.y_train)
        m.fit(split.X_train, split.y_train,
              eval_set=[(split.X_valid, split.y_valid)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        raise ValueError(f"unknown model_type {model_type!r}")
    return m


def suggest(trial, model_type: str) -> dict:
    if model_type == "xgboost":
        return {
            "n_estimators": 1500,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        }
    return {
        "n_estimators": 2000,
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
    }


# --------------------------------------------------------------------------
# tuning
# --------------------------------------------------------------------------
def tune_optuna(model_type: str, split: Split, n_trials: int) -> tuple[dict, float, float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = suggest(trial, model_type)
        m = fit_model(model_type, params, split)
        return average_precision_score(
            split.y_valid, m.predict_proba(split.X_valid)[:, 1]
        )

    t0 = time.perf_counter()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    log.info("[%s] optuna: %d trials in %.1fs, best valid PR-AUC=%.5f",
             model_type, n_trials, elapsed, study.best_value)
    return study.best_params, float(study.best_value), elapsed


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def run_training(
    *,
    n_trials: int = 30,
    model_types: list[str] | None = None,
    use_ray: bool = False,
    register: bool = True,
) -> dict:
    import mlflow

    model_types = model_types or ["xgboost", "lightgbm"]
    df = load_or_build()
    split = chronological_split(df)
    base = baselines(split)
    log.info("baselines: %s", base)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    results: list[ModelResult] = []
    best: tuple[float, Any, ModelResult] | None = None

    for mt in model_types:
        with mlflow.start_run(run_name=f"{mt}-{'ray' if use_ray else 'optuna'}"):
            mlflow.log_params({"model_type": mt, "n_trials": n_trials,
                               "tuner": "ray" if use_ray else "optuna",
                               "n_features": len(ALL_MODEL_INPUTS)})
            mlflow.log_params({f"split_{k}": v for k, v in split.describe().items()})
            mlflow.log_metrics(base)

            if use_ray:
                from fraudpulse.training.tune_ray import tune_ray

                params, valid_ap, tune_s = tune_ray(mt, split, n_trials)
            else:
                params, valid_ap, tune_s = tune_optuna(mt, split, n_trials)

            t0 = time.perf_counter()
            model = fit_model(mt, params, split)
            fit_s = time.perf_counter() - t0

            test_scores = evaluate(
                split.y_test.to_numpy(), model.predict_proba(split.X_test)[:, 1]
            )
            res = ModelResult(
                model_type=mt,
                params=params,
                valid_pr_auc=valid_ap,
                test_pr_auc=test_scores["pr_auc"],
                test_roc_auc=test_scores["roc_auc"],
                test_precision_at_recall_50=test_scores["precision_at_recall_50"],
                test_recall_at_precision_90=test_scores["recall_at_precision_90"],
                fit_seconds=fit_s,
                n_trials=n_trials,
                tuning_seconds=tune_s,
            )
            results.append(res)

            mlflow.log_params({f"best_{k}": v for k, v in params.items()})
            mlflow.log_metrics({
                "valid_pr_auc": valid_ap,
                "test_pr_auc": res.test_pr_auc,
                "test_roc_auc": res.test_roc_auc,
                "test_precision_at_recall_50": res.test_precision_at_recall_50,
                "test_recall_at_precision_90": res.test_recall_at_precision_90,
                "tuning_seconds": tune_s,
                "fit_seconds": fit_s,
                "pr_auc_lift_over_amount_only":
                    res.test_pr_auc - base["baseline_amount_only_pr_auc"],
            })
            _log_importance(model, mt)

            log.info(
                "[%s] valid PR-AUC=%.5f  test PR-AUC=%.5f  (amount-only baseline %.5f, "
                "prevalence %.5f)",
                mt, valid_ap, res.test_pr_auc,
                base["baseline_amount_only_pr_auc"], base["baseline_prevalence"],
            )
            if best is None or valid_ap > best[0]:
                best = (valid_ap, model, res)

    assert best is not None
    _, best_model, best_res = best
    log.info("winner: %s (valid PR-AUC=%.5f)", best_res.model_type, best_res.valid_pr_auc)

    if register:
        _register(best_model, best_res, split, base)

    summary = {
        "baselines": base,
        "results": [asdict(r) for r in results],
        "winner": best_res.model_type,
        "split": split.describe(),
    }
    out = settings.reports_dir / "training_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out)
    return summary


def _log_importance(model, model_type: str) -> None:
    import mlflow

    try:
        imp = pd.Series(model.feature_importances_, index=ALL_MODEL_INPUTS)
        imp = imp.sort_values(ascending=False)
        mlflow.log_text(imp.to_string(), f"feature_importance_{model_type}.txt")
        log.info("top features (%s): %s", model_type,
                 ", ".join(f"{k}={v:.4f}" for k, v in imp.head(8).items()))
    except Exception as exc:  # importance is nice-to-have, never fatal
        log.warning("could not log feature importance: %s", exc)


def _register(model, res: ModelResult, split: Split, base: dict) -> None:
    """Register the winner and persist everything serving needs alongside it."""
    import mlflow

    with mlflow.start_run(run_name=f"register-{res.model_type}"):
        mlflow.log_params({"model_type": res.model_type, **{f"p_{k}": v
                                                            for k, v in res.params.items()}})
        mlflow.log_metrics({"test_pr_auc": res.test_pr_auc, "valid_pr_auc": res.valid_pr_auc,
                            **base})
        signature = mlflow.models.infer_signature(
            split.X_test.head(100), model.predict_proba(split.X_test.head(100))[:, 1]
        )
        flavour = mlflow.xgboost if res.model_type == "xgboost" else mlflow.lightgbm
        flavour.log_model(
            model,
            name="model",
            signature=signature,
            input_example=split.X_test.head(5),
            registered_model_name=settings.registered_model_name,
        )
        mlflow.log_dict(
            {
                "feature_order": ALL_MODEL_INPUTS,
                "model_type": res.model_type,
                "categories": split.categories,
                "threshold": settings.score_threshold,
                "test_pr_auc": res.test_pr_auc,
            },
            "serving_metadata.json",
        )
    log.info("registered %s as '%s'", res.model_type, settings.registered_model_name)
