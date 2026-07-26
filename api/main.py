"""FastAPI inference service for the bilingual hate-speech classifier.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Model is chosen by env HSC_MODEL_ID, else the best test-macro-F1 model in the registry.

The response is probabilistic and NOT a moderation verdict (see the /health disclaimer).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from hsc.inference import get_classifier

_MODEL_ID = os.environ.get("HSC_MODEL_ID") or None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm up: load model + language detector once at startup
    clf = get_classifier(_MODEL_ID)
    clf.predict("warmup")
    app.state.clf = clf
    yield


app = FastAPI(
    title="Bilingual Hate-Speech Classifier",
    version="0.1.0",
    description="EN/PT hate vs not-hate. Probabilistic research tool, not a moderation oracle.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=app.state.clf.model_id)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    pred = app.state.clf.predict(req.text)
    pred["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return PredictResponse(**pred)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    t0 = time.perf_counter()
    preds = app.state.clf.predict_batch(req.texts)
    return BatchResponse(
        predictions=preds, latency_ms=round((time.perf_counter() - t0) * 1000, 2)
    )
