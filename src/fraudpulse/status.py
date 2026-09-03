"""`fraudpulse status` — a one-screen answer to "what actually exists right now?"."""

from __future__ import annotations

import socket
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fraudpulse.config import settings

console = Console()


def _port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _file_row(label: str, path: Path) -> tuple[str, str, str]:
    if not path.exists():
        return (label, "[red]missing[/red]", str(path))
    if path.is_dir():
        files = list(path.glob("*.parquet"))
        size = sum(f.stat().st_size for f in files)
        return (label, f"[green]{len(files)} files[/green]", f"{size / 1e6:.1f} MB")
    return (label, "[green]ok[/green]", f"{path.stat().st_size / 1e6:.1f} MB")


def print_status() -> None:
    svc = Table(title="services", show_header=True, header_style="bold")
    svc.add_column("service")
    svc.add_column("endpoint")
    svc.add_column("state")
    for name, host, port in [
        (
            "kafka (redpanda)",
            settings.kafka_bootstrap.split(":")[0],
            int(settings.kafka_bootstrap.split(":")[1]),
        ),
        ("redis", settings.redis_host, settings.redis_port),
        ("mlflow", "localhost", int(settings.mlflow_tracking_uri.rsplit(":", 1)[1])),
        ("api", "localhost", 8000),
    ]:
        up = _port_open(host, port)
        svc.add_row(name, f"{host}:{port}", "[green]up[/green]" if up else "[red]down[/red]")
    console.print(svc)

    art = Table(title="artifacts", show_header=True, header_style="bold")
    art.add_column("artifact")
    art.add_column("state")
    art.add_column("detail")
    for label, path in [
        ("raw csv", settings.raw_dir / "train_transaction.csv"),
        ("events.parquet", settings.processed_dir / "events.parquet"),
        ("landing zone", settings.landing_dir),
        ("offline features", settings.processed_dir / "offline_features.parquet"),
        ("feast registry", settings.feature_repo_dir / "data" / "registry.db"),
        ("feast offline parquet", settings.feature_repo_dir / "data" / "card_features.parquet"),
        ("training set", settings.processed_dir / "training_set.parquet"),
        ("parity report", settings.reports_dir / "parity_report.json"),
        ("latency report", settings.reports_dir / "latency.json"),
    ]:
        art.add_row(*_file_row(label, path))
    console.print(art)
