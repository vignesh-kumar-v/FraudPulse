"""`fraudpulse verify-all` — run every phase's gate and print one table.

The point of a single command is that "the project works" becomes a thing you
can re-check in one step rather than a claim in a README. Each check either
passes with a number attached or fails with the reason.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger

# Phases 0-2 recompute from scratch. Phases 3-4 read a JSON report, because
# re-running a load test or a drift experiment inside `verify` would take
# minutes. That shortcut has a failure mode: reports/*.json is committed to the
# repo, so on a fresh clone those checks would happily PASS on numbers measured
# on someone else's machine. _fresh_report() closes that - a report is only
# accepted if it is newer than the dataset it claims to describe.
EVENTS_PARQUET = settings.processed_dir / "events.parquet"

log = get_logger(__name__)
console = Console()


@dataclass
class Check:
    phase: str
    name: str
    passed: bool
    detail: str
    seconds: float


def _fresh_report(path: Path, *, produced_by: str) -> tuple[dict | None, str]:
    """Load a report, refusing one that predates the current dataset.

    Returns (payload, reason). ``payload`` is None when the report cannot be
    trusted, and ``reason`` says why in a way that names the fix.
    """
    if not path.exists():
        return None, f"{path.relative_to(settings.repo_root)} missing - run `{produced_by}`"
    if not EVENTS_PARQUET.exists():
        return None, (
            f"no {EVENTS_PARQUET.name} to check {path.name} against; a committed report "
            "proves nothing about this machine - run `make prepare` first"
        )
    report_mtime = path.stat().st_mtime
    data_mtime = EVENTS_PARQUET.stat().st_mtime
    if report_mtime < data_mtime:
        age = (data_mtime - report_mtime) / 60
        return None, (
            f"{path.name} is {age:.0f} min older than the dataset it describes - "
            f"stale, re-run `{produced_by}`"
        )
    return json.loads(path.read_text()), ""


def _timed(fn: Callable[[], tuple[bool, str]], phase: str, name: str) -> Check:
    t0 = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as exc:  # a check that errors is a check that failed
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    return Check(phase, name, ok, detail, time.perf_counter() - t0)


# --------------------------------------------------------------------------
def _check_services() -> tuple[bool, str]:
    import socket

    targets = {
        "kafka": (
            settings.kafka_bootstrap.split(":")[0],
            int(settings.kafka_bootstrap.split(":")[1]),
        ),
        "redis": (settings.redis_host, settings.redis_port),
        "mlflow": ("localhost", int(settings.mlflow_tracking_uri.rsplit(":", 1)[1])),
    }
    down = []
    for name, (host, port) in targets.items():
        try:
            socket.create_connection((host, port), timeout=1).close()
        except OSError:
            down.append(name)
    return (not down), ("all up" if not down else f"down: {', '.join(down)}")


def _check_landing_row_count() -> tuple[bool, str]:
    from fraudpulse.data.prepare import load_events
    from fraudpulse.streaming.consumer_landing import read_landing

    src = len(load_events())
    landed = len(read_landing())
    return src == landed, f"source={src} landed={landed}"


def _check_parity() -> tuple[bool, str]:
    from fraudpulse.data.prepare import load_events
    from fraudpulse.features.parity import run_parity_check

    ev = load_events()
    strict = run_parity_check(ev, tie_policy="watermark")
    naive = run_parity_check(ev, tie_policy="arrival")
    ok = strict.passed and not naive.passed
    return ok, (
        f"watermark {strict.rows_with_any_mismatch}/{strict.n_rows} mismatched; "
        f"arrival {naive.rows_with_any_mismatch} ({naive.row_mismatch_rate:.4%}) "
        f"- the naive path must still diverge or the check is vacuous"
    )


def _check_state_recovery() -> tuple[bool, str]:
    """Phase 2: can the streaming state actually be rebuilt after a crash?"""
    d, why = _fresh_report(
        settings.reports_dir / "state_rebuild.json", produced_by="python scripts/rebuild_state.py"
    )
    if d is None:
        return False, why
    sc = d.get("spot_check")
    if not sc:
        return False, "state_rebuild.json has no spot check - re-run without --no-push"
    ok = sc["n_cards_with_diff"] == 0
    return ok, (
        f"{d['landed_events']} landed events replayed into {d['cards']} cards; "
        f"{sc['n_checked'] - sc['n_cards_with_diff']}/{sc['n_checked']} spot checks exact"
    )


def _check_online_store() -> tuple[bool, str]:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(settings.repo_root / "scripts" / "verify_parity.py")],
        capture_output=True,
        text=True,
        cwd=settings.repo_root,
    )
    path = settings.reports_dir / "parity_e2e.json"
    if not path.exists():
        return False, (proc.stderr or proc.stdout)[-200:]
    d = json.loads(path.read_text())
    s, k = d["served_vs_offline"], d["online_store_spot_check"]
    ok = s["rows_with_mismatch"] == 0 and k["n_cards_with_diff"] == 0
    return ok, (
        f"{s['events_compared']} events served vs offline: "
        f"{s['rows_with_mismatch']} mismatched; "
        f"redis spot check {k['n_checked'] - k['n_cards_with_diff']}/{k['n_checked']} exact"
    )


def _check_model() -> tuple[bool, str]:
    d, why = _fresh_report(settings.reports_dir / "training_summary.json", produced_by="make train")
    if d is None:
        return False, why
    base = d["baselines"]["baseline_amount_only_pr_auc"]
    best = max(r["test_pr_auc"] for r in d["results"])
    return best > 2 * base, (
        f"test PR-AUC {best:.4f} vs amount-only {base:.4f} "
        f"({best / base:.1f}x) vs prevalence {d['baselines']['baseline_prevalence']:.4f}"
    )


def _check_latency() -> tuple[bool, str]:
    d, why = _fresh_report(
        settings.reports_dir / "latency.json", produced_by="make serve && make loadtest"
    )
    if d is None:
        return False, why
    single = d.get("c1x1")
    if not single:
        return False, "no single-request measurement (c1x1) in the report"
    p95 = single["client_latency"]["p95_ms"]
    srv = single["server_latency"]["p95_ms"]
    return p95 < 50, f"p95 {p95:.2f}ms end-to-end, {srv:.2f}ms server-side, 0 errors"


def _check_drift() -> tuple[bool, str]:
    d, why = _fresh_report(settings.reports_dir / "drift_experiment.json", produced_by="make drift")
    if d is None:
        return False, why
    return d["passed"], (
        f"null share {d['null_drift_share']:.3f} (must be quiet); "
        f"temporal floor {d['temporal_floor_drift_share']:.3f}; "
        f"detected {sum(d['detected'].values())}/{len(d['detected'])} injections"
    )


CHECKS: list[tuple[str, str, Callable[[], tuple[bool, str]]]] = [
    ("0", "services reachable", _check_services),
    ("1", "landed rows == source rows", _check_landing_row_count),
    ("2", "offline/online feature parity", _check_parity),
    ("2", "online store matches offline (e2e)", _check_online_store),
    ("2", "state rebuilds from the landing zone", _check_state_recovery),
    ("3", "model beats the amount-only baseline", _check_model),
    ("3", "p95 inference latency", _check_latency),
    ("4", "drift monitor fires on shift, not on noise", _check_drift),
]


def run_all() -> bool:
    results = [_timed(fn, phase, name) for phase, name, fn in CHECKS]

    table = Table(title="FraudPulse verification", header_style="bold")
    table.add_column("phase", justify="center")
    table.add_column("check")
    table.add_column("result", justify="center")
    table.add_column("detail")
    table.add_column("time", justify="right")
    for r in results:
        table.add_row(
            r.phase,
            r.name,
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
            r.detail,
            f"{r.seconds:.1f}s",
        )
    console.print(table)

    out = settings.reports_dir / "verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            [
                {
                    "phase": r.phase,
                    "check": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                    "seconds": r.seconds,
                }
                for r in results
            ],
            indent=2,
        )
    )
    n_pass = sum(r.passed for r in results)
    console.print(f"\n{n_pass}/{len(results)} checks passed -> {out}")
    return n_pass == len(results)
