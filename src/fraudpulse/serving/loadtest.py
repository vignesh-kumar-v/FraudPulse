"""Latency measurement for the scoring API.

Deliberately not a throughput benchmark. The number this project claims is p95
*inference latency*, so the load generator is asyncio + httpx over persistent
connections, sampling real card IDs so the online store is actually hit (a
benchmark that only ever asks for missing cards measures the cold-start path and
reports a flatteringly small number).

Reported: p50, p90, p95, p99, max, plus the server's own self-timing so client
overhead and network time can be separated from work the service actually did.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np

from fraudpulse.config import settings
from fraudpulse.logging_utils import get_logger

log = get_logger(__name__)


def _sample_cards(n: int) -> list[dict]:
    """Real (card, amount, product) triples so the store lookups actually hit."""
    import pandas as pd

    path = settings.processed_dir / "events.parquet"
    if not path.exists():
        log.warning("%s missing; load-testing with synthetic card ids (cold-start path "
                    "only - the numbers will be optimistic)", path)
        return [{"card_id": f"card_{i}", "amount": 100.0, "product_cd": "W"} for i in range(n)]
    df = pd.read_parquet(path, columns=["card_id", "amount", "product_cd", "card_network",
                                        "card_type", "email_domain", "addr1", "dist1"])
    sample = df.sample(n=min(n, len(df)), random_state=0)
    return sample.to_dict("records")


def percentiles(xs: list[float]) -> dict[str, float]:
    a = np.array(xs)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }


async def _worker(client, queue, url, explain, client_ms, server_ms, errors):
    while True:
        try:
            payload = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/score", json={**payload, "explain": explain})
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                errors.append(r.status_code)
                continue
            client_ms.append(elapsed)
            server_ms.append(float(r.json()["latency_ms"]))
        except Exception as exc:  # noqa: BLE001 - a failed request is data, not a crash
            errors.append(str(exc)[:80])


async def _run(n: int, concurrency: int, base_url: str, explain: bool) -> dict:
    rows = _sample_cards(n)
    queue: asyncio.Queue = asyncio.Queue()
    for i, r in enumerate(rows):
        queue.put_nowait({
            "transaction_id": i,
            "card_id": str(r["card_id"]),
            "amount": float(r["amount"]),
            "product_cd": str(r.get("product_cd", "W")),
            "card_network": str(r.get("card_network", "unknown")),
            "card_type": str(r.get("card_type", "unknown")),
            "email_domain": str(r.get("email_domain", "unknown")),
            "addr1": None if r.get("addr1") is None else float(r["addr1"]),
            "dist1": None if r.get("dist1") is None else float(r["dist1"]),
        })

    client_ms: list[float] = []
    server_ms: list[float] = []
    errors: list = []

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        # warm the connection pool so TCP/TLS setup does not land in the sample
        await client.get(f"{base_url}/health")
        t0 = time.perf_counter()
        await asyncio.gather(*[
            _worker(client, queue, base_url, explain, client_ms, server_ms, errors)
            for _ in range(concurrency)
        ])
        wall = time.perf_counter() - t0

    return {
        "requests": n,
        "concurrency": concurrency,
        "explain": explain,
        "wall_seconds": wall,
        "throughput_rps": len(client_ms) / wall if wall else 0.0,
        "errors": len(errors),
        "error_sample": errors[:5],
        "client_latency": percentiles(client_ms) if client_ms else {},
        "server_latency": percentiles(server_ms) if server_ms else {},
    }


def run_loadtest(
    *, n: int = 2_000, concurrency: int = 16,
    base_url: str = "http://localhost:8000", explain: bool = False,
    out_path: Path | None = None,
) -> dict:
    random.seed(0)
    result = asyncio.run(_run(n, concurrency, base_url, explain))

    c, s = result["client_latency"], result["server_latency"]
    log.info(
        "%d req @ concurrency=%d explain=%s -> %.0f rps, %d errors",
        n, concurrency, explain, result["throughput_rps"], result["errors"],
    )
    if c:
        log.info("client p50=%.2fms p95=%.2fms p99=%.2fms max=%.2fms",
                 c["p50_ms"], c["p95_ms"], c["p99_ms"], c["max_ms"])
        log.info("server p50=%.2fms p95=%.2fms p99=%.2fms max=%.2fms",
                 s["p50_ms"], s["p95_ms"], s["p99_ms"], s["max_ms"])

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(out_path.read_text()) if out_path.exists() else {}
        existing["shap" if explain else "baseline"] = result
        out_path.write_text(json.dumps(existing, indent=2))
        log.info("wrote %s", out_path)
    return result
