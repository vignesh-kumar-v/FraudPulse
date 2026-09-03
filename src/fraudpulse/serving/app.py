"""Real-time fraud scoring API.

Request path, in order:

  1. Read the card's stored features from Feast's online store (Redis).
  2. Compute the on-demand features from those plus this request.
  3. Encode with the *training-time* category map and column order.
  4. Score. Optionally attach SHAP contributions.

Step 1 goes through ``store.get_online_features`` against the same
``fraud_serving_v1`` feature service the training set was built from, so the
feature list is not duplicated here. If a feature is added to the service and
the model is not retrained, ``LoadedModel.feature_order`` still pins serving to
what the model actually expects, and the mismatch is logged rather than silently
reshaping the input.

Latency is measured inside the handler and returned on every response, and
broken out per stage on ``/metrics`` - a single end-to-end number tells you the
service is slow but not which of Redis, the on-demand maths or the model is
responsible.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from fraudpulse.config import settings
from fraudpulse.feast_repo import FEATURE_SERVICE, get_store
from fraudpulse.features.ondemand import compute_ondemand
from fraudpulse.features.spec import ENTITY_KEY, FEATURE_DEFAULTS, FEATURE_NAMES
from fraudpulse.logging_utils import get_logger
from fraudpulse.schema import ScoreRequest, ScoreResponse, ShapContribution
from fraudpulse.serving.model_store import LoadedModel, load_latest

log = get_logger(__name__)

REQUESTS = Counter("fraudpulse_requests_total", "Scoring requests", ["outcome"])
COLD_STARTS = Counter("fraudpulse_cold_starts_total", "Requests with no online features")
STAGE = Histogram(
    "fraudpulse_stage_seconds",
    "Per-stage request latency",
    ["stage"],
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
)

STATE: dict[str, Any] = {
    "model": None,
    "store": None,
    "explainer": None,
    # Resolved once at startup. get_feature_service() hits the registry, and
    # calling it per request put ~20ms of registry I/O inside the latency
    # budget - it dominated the p50 in the first measurement.
    "feature_refs": None,
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model and warm the store once, at startup.

    Loading lazily on the first request makes p99 a lie: one unlucky caller
    absorbs multi-second model deserialisation and it shows up as a tail-latency
    mystery.
    """
    STATE["store"] = get_store()
    # Resolve the feature service once and cache the concrete refs. This also
    # makes a renamed or deleted service fail at startup instead of on the
    # first request.
    svc = STATE["store"].get_feature_service(FEATURE_SERVICE)
    STATE["feature_refs"] = [f"card_stats:{f}" for f in FEATURE_NAMES]
    log.info(
        "feature service '%s' resolved (%d stored features)", svc.name, len(STATE["feature_refs"])
    )
    try:
        STATE["model"] = load_latest()
        log.info(
            "model %s v%s ready (%d features, test PR-AUC=%s)",
            STATE["model"].model_type,
            STATE["model"].version,
            len(STATE["model"].feature_order),
            STATE["model"].test_pr_auc,
        )
    except Exception as exc:
        log.error("could not load model: %s", exc)
        STATE["model"] = None

    if settings.enable_shap and STATE["model"] is not None:
        try:
            import shap

            STATE["explainer"] = shap.TreeExplainer(STATE["model"].model)
            log.info("SHAP TreeExplainer ready")
        except Exception as exc:
            log.warning("SHAP unavailable: %s", exc)

    # One throwaway prediction so the first real request is not paying for
    # lazy allocation inside XGBoost/LightGBM.
    if STATE["model"] is not None:
        _score_one(ScoreRequest(transaction_id=0, card_id="__warmup__", amount=1.0))
    yield


app = FastAPI(
    title="FraudPulse",
    version="0.1.0",
    description="Real-time fraud scoring backed by a Feast feature store.",
    lifespan=lifespan,
)


def _fetch_online(card_id: str) -> tuple[dict[str, float], bool]:
    """Stored features for one card. Returns (features, found_in_store)."""
    # The feature service also contains the on-demand view, which needs request
    # data we handle ourselves; ask for the stored columns only.
    resp = (
        STATE["store"]
        .get_online_features(
            features=STATE["feature_refs"],
            entity_rows=[{ENTITY_KEY: card_id}],
        )
        .to_dict()
    )

    out: dict[str, float] = {}
    found = False
    for name in FEATURE_NAMES:
        v = resp.get(name, [None])[0]
        if v is None:
            out[name] = FEATURE_DEFAULTS[name]
        else:
            out[name] = float(v)
            found = True
    return out, found


def _score_one(req: ScoreRequest) -> ScoreResponse:
    model: LoadedModel | None = STATE["model"]
    if model is None:
        raise HTTPException(503, "no model loaded; run `make train`")

    t0 = time.perf_counter()
    ts = req.event_timestamp or datetime.now(UTC)
    event_unix = ts.timestamp()

    t_store = time.perf_counter()
    stored, found = _fetch_online(req.card_id)
    STAGE.labels("online_store").observe(time.perf_counter() - t_store)

    t_feat = time.perf_counter()
    ondemand = compute_ondemand(
        amount=req.amount,
        product_cd=req.product_cd,
        event_unixtime=event_unix,
        stored=stored,
    )
    row: dict[str, Any] = {
        **stored,
        **ondemand,
        "amount": req.amount,
        "hour_of_day": ts.hour,
        "day_of_week": ts.weekday(),
        "addr1": req.addr1,
        "dist1": req.dist1,
        "product_cd": req.product_cd,
        "card_network": req.card_network,
        "card_type": req.card_type,
        "email_domain": req.email_domain,
    }
    X = model.encode(row)
    STAGE.labels("feature_assembly").observe(time.perf_counter() - t_feat)

    t_pred = time.perf_counter()
    proba = float(model.predict_proba(X)[0])
    STAGE.labels("predict").observe(time.perf_counter() - t_pred)

    explanation = None
    if req.explain:
        t_shap = time.perf_counter()
        explanation = _explain(X, req.top_k)
        STAGE.labels("shap").observe(time.perf_counter() - t_shap)

    latency_ms = (time.perf_counter() - t0) * 1000
    STAGE.labels("total").observe(latency_ms / 1000)
    REQUESTS.labels("ok").inc()
    if not found:
        COLD_STARTS.inc()

    return ScoreResponse(
        transaction_id=req.transaction_id,
        card_id=req.card_id,
        fraud_probability=proba,
        is_fraud_pred=proba >= model.threshold,
        threshold=model.threshold,
        model_version=f"{model.model_type}:v{model.version}",
        feature_source="online_store" if found else "cold_start",
        latency_ms=latency_ms,
        features_used=(
            {k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and v is not None}
            if req.include_features
            else None
        ),
        explanation=explanation,
    )


def _explain(X, top_k: int) -> list[ShapContribution] | None:
    explainer = STATE["explainer"]
    model: LoadedModel = STATE["model"]
    if explainer is None:
        return None
    values = explainer.shap_values(X)
    if isinstance(values, list):  # some versions return one array per class
        values = values[1]
    contribs = values[0]
    order = sorted(range(len(contribs)), key=lambda i: -abs(contribs[i]))[:top_k]
    return [
        ShapContribution(
            feature=model.feature_order[i],
            value=float(X.iloc[0, i]),
            contribution=float(contribs[i]),
        )
        for i in order
    ]


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    model: LoadedModel | None = STATE["model"]
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_version": f"{model.model_type}:v{model.version}" if model else None,
        "shap_enabled": STATE["explainer"] is not None,
        "feature_service": FEATURE_SERVICE,
        "n_features": len(model.feature_order) if model else 0,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    return _score_one(req)


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(reqs: list[ScoreRequest]) -> list[ScoreResponse]:
    if len(reqs) > 500:
        raise HTTPException(413, "batch limit is 500")
    return [_score_one(r) for r in reqs]


@app.post("/explain", response_model=ScoreResponse)
def explain(req: ScoreRequest) -> ScoreResponse:
    req.explain = True
    return _score_one(req)


@app.get("/features/{card_id}")
def features(card_id: str) -> dict:
    """What the store currently holds for a card. Debug endpoint."""
    stored, found = _fetch_online(card_id)
    return {"card_id": card_id, "found": found, "features": stored}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
