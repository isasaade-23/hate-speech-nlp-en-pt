"""FastAPI inference service for the bilingual hate-speech classifier.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Model is chosen by env HSC_MODEL_ID, else the best test-macro-F1 model in the registry.
Access: header X-API-Key (see api/security.py). Routes live under /v1; /health is open.

The response is probabilistic and NOT a moderation verdict (see the /health notice).
Logs carry no input text: only client name, counts, language and latency.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from api.security import autenticar, gastar
from hsc.inference import get_classifier

_MODEL_ID = os.environ.get("HSC_MODEL_ID") or None
log = logging.getLogger("hsc.api")

NOTICE = (
    "Probabilistic research tool, not a moderation verdict. Research and evaluation use only; "
    "do not use scores for automated decisions about people without human review."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm up: load model + language detector once at startup
    clf = get_classifier(_MODEL_ID)
    clf.predict("warmup")
    app.state.clf = clf
    yield


app = FastAPI(
    title="Bilingual Hate-Speech Classifier",
    version="0.2.0",
    description="EN/PT hate vs not-hate. " + NOTICE,
    lifespan=lifespan,
)


def _erro(status: int, code: str, message: str | None = None, headers: dict | None = None):
    return JSONResponse({"error": {"code": code, "message": message or code}},
                        status_code=status, headers=headers)


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException):
    return _erro(exc.status_code, str(exc.detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    # so campo e tipo do erro; a entrada do usuario nao volta na resposta nem vai para o log
    campos = [".".join(str(p) for p in e["loc"] if p != "body") for e in exc.errors()]
    return _erro(422, "invalid_request", "invalid fields: " + ", ".join(campos))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=app.state.clf.model_id, notice=NOTICE)


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest, cliente: str = Depends(autenticar)) -> PredictResponse:
    gastar(cliente, 1)
    t0 = time.perf_counter()
    pred = app.state.clf.predict(req.text)
    pred["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    log.info("predict client=%s chars=%d lang=%s ms=%.1f", cliente, len(req.text),
             pred["language"]["detected"], pred["latency_ms"])
    return PredictResponse(**pred)


@app.post("/v1/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest, cliente: str = Depends(autenticar)) -> BatchResponse:
    gastar(cliente, len(req.texts))
    t0 = time.perf_counter()
    preds = app.state.clf.predict_batch(req.texts)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info("predict_batch client=%s n=%d chars=%d ms=%.1f", cliente, len(req.texts),
             sum(len(t) for t in req.texts), ms)
    return BatchResponse(predictions=preds, latency_ms=ms)
