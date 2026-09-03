"""FraudPulse command line — one subcommand per build phase.

Every ``make`` target is a thin call into here, so the phases are runnable and
inspectable individually rather than only as one opaque pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger

app = typer.Typer(add_completion=False, help="FraudPulse — real-time fraud detection.")
log = get_logger("cli")


# ------------------------------------------------------------------ phase 0/1
@app.command()
def prepare(
    sample_cards: int = typer.Option(0, help="Keep only N random cards (0 = all)."),
    synthetic: bool = typer.Option(False, help="Generate synthetic events instead of IEEE-CIS."),
    n_events: int = typer.Option(200_000, help="Synthetic mode only."),
) -> None:
    """Build data/processed/events.parquet."""
    from fraudpulse.data.prepare import prepare_events

    settings.ensure_dirs()
    if synthetic:
        from fraudpulse.data.synthetic import make_events

        df = make_events(n_cards=max(n_events // 40, 50), n_events=n_events)
        out = settings.processed_dir / "events.parquet"
        df.to_parquet(out, index=False)
        log.info("wrote synthetic events -> %s (%d rows)", out, len(df))
    else:
        prepare_events(sample_cards=sample_cards or None)


@app.command()
def topic(recreate: bool = typer.Option(False, help="Delete and recreate the topic.")) -> None:
    """Create the transactions topic."""
    from fraudpulse.streaming.topics import delete_topic, ensure_topic

    if recreate:
        delete_topic()
        import time

        time.sleep(2)
    ensure_topic()


@app.command()
def produce(
    speedup: float = typer.Option(0.0, help="Replay pace vs. data clock; 0 = as fast as possible."),
    max_events: int = typer.Option(0, help="Cap events produced (0 = all)."),
) -> None:
    """PHASE 1 — replay the dataset onto Kafka."""
    from fraudpulse.data.prepare import load_events
    from fraudpulse.streaming.producer import replay

    events = load_events()
    stats = replay(events, speedup=speedup, max_events=max_events or None)
    raise typer.Exit(0 if stats.failed == 0 and stats.delivered == stats.attempted else 1)


@app.command()
def land(
    expect: int = typer.Option(0, help="Expected row count; logs an error on mismatch."),
    idle_timeout: float = typer.Option(20.0),
) -> None:
    """PHASE 1 — consume raw events into the Parquet landing zone."""
    from fraudpulse.streaming import consumer_landing

    consumer_landing.run(expect=expect or None, idle_timeout_s=idle_timeout)


# -------------------------------------------------------------------- phase 2
@app.command()
def build_offline() -> None:
    """PHASE 2 — compute offline features and write the Feast FileSource parquet."""
    from fraudpulse.data.prepare import load_events
    from fraudpulse.feast_repo import apply_repo, write_offline_parquet
    from fraudpulse.features.offline import compute_offline_features

    events = load_events()
    feats = compute_offline_features(events)
    write_offline_parquet(feats)
    feats.to_parquet(settings.processed_dir / "offline_features.parquet", index=False)
    apply_repo()


@app.command()
def features(
    tie_policy: str = typer.Option("watermark", help="watermark | arrival"),
    capture: bool = typer.Option(True, help="Capture the served scoring view for parity."),
    idle_timeout: float = typer.Option(20.0),
) -> None:
    """PHASE 2 — stream events, compute online features, push to Redis via Feast."""
    from fraudpulse.streaming import consumer_features

    capture_path = settings.processed_dir / "online_scoring_view.parquet" if capture else None
    consumer_features.run(
        tie_policy=tie_policy,  # type: ignore[arg-type]
        capture_scoring_view=capture_path,
        idle_timeout_s=idle_timeout,
    )


@app.command()
def parity(
    tie_policy: str = typer.Option("both", help="watermark | arrival | both"),
    out: Path = typer.Option(Path("reports/parity_report.json")),
) -> None:
    """PHASE 2 verify — offline vs. online feature parity."""
    from fraudpulse.data.prepare import load_events
    from fraudpulse.features.parity import run_parity_check, write_report

    events = load_events()
    policies = ["arrival", "watermark"] if tie_policy == "both" else [tie_policy]
    reports = []
    for pol in policies:
        rep = run_parity_check(events, tie_policy=pol)  # type: ignore[arg-type]
        typer.echo("=" * 78)
        typer.echo(rep.summary())
        reports.append(rep)
    write_report(reports, settings.repo_root / out)
    raise typer.Exit(0 if reports[-1].passed else 1)


# -------------------------------------------------------------------- phase 3
@app.command()
def train(
    trials: int = typer.Option(30, help="Optuna trials."),
    models: str = typer.Option("xgboost,lightgbm"),
    ray: bool = typer.Option(False, help="Use Ray Tune instead of sequential Optuna."),
) -> None:
    """PHASE 3 — build the training set, tune, train, register in MLflow."""
    from fraudpulse.training.train import run_training

    run_training(n_trials=trials, model_types=models.split(","), use_ray=ray)


@app.command()
def loadtest(
    n: int = typer.Option(2_000),
    concurrency: int = typer.Option(16, help="Concurrent requests per client process."),
    processes: int = typer.Option(
        1, help="Client processes. >1 avoids the load generator becoming the bottleneck."
    ),
    url: str = typer.Option("http://localhost:8000"),
    explain: bool = typer.Option(False, help="Measure the SHAP endpoint's added latency."),
    out: Path = typer.Option(Path("reports/latency.json")),
) -> None:
    """PHASE 3 verify — measure p50/p95/p99 inference latency."""
    from fraudpulse.serving.loadtest import run_loadtest

    run_loadtest(
        n=n, concurrency=concurrency, processes=processes, base_url=url, explain=explain,
        out_path=settings.repo_root / out,
    )


# -------------------------------------------------------------------- phase 4
@app.command()
def drift(
    inject: str = typer.Option("all", help="all | amount | velocity | product"),
    out: Path = typer.Option(Path("reports")),
) -> None:
    """PHASE 4 verify — run the drift monitor against an injected shift."""
    from fraudpulse.monitoring.drift import run_drift_experiment

    result = run_drift_experiment(inject=inject, out_dir=settings.repo_root / out)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def hpo_compare(
    trials: int = typer.Option(20),
    model: str = typer.Option("xgboost"),
) -> None:
    """PHASE 5 verify (stretch) — sequential Optuna vs. parallel Ray Tune wall-clock."""
    from fraudpulse.training.tune_ray import compare_sequential_vs_parallel

    typer.echo(json.dumps(compare_sequential_vs_parallel(model, trials), indent=2))


@app.command()
def verify_all() -> None:
    """Run every phase's verification check and print a pass/fail table."""
    from fraudpulse.verify import run_all

    raise typer.Exit(0 if run_all() else 1)


@app.command()
def status() -> None:
    """Print what exists so far: services, data files, registry, model."""
    from fraudpulse.status import print_status

    print_status()


if __name__ == "__main__":
    app()
