# API and product

## What the API is

The API is the product form of the model: a small web server (FastAPI) that other software
talks to over HTTP. Instead of running Python, an app, website, or bot sends a piece of text
and receives a structured answer.

```http
POST /v1/predict          (header X-API-Key: <chave>)
{"text": "..."}
```

```json
{
  "text": "...",
  "label": "hate",
  "score": 0.74,
  "language": {"detected": "pt", "confidence": 0.86},
  "model_version": "tfidf_logreg_strict_s42",
  "latency_ms": 35
}
```

Endpoints:

- `GET /health`: liveness, the served model version and the use notice. Open, no key.
- `POST /v1/predict`: one text (up to 2,000 characters).
- `POST /v1/predict_batch`: up to 256 texts at once.

Access and limits:

- Every `/v1` call needs the header `X-API-Key`. The server stores only the SHA-256 of each key
  (`HSC_API_KEYS="name:sha256,..."`); generate one with `python -m api.security NAME`.
  `HSC_API_AUTH=off` disables the check for local development only.
- Usage limit per key: `HSC_RATE_PER_MIN` units per minute (default 600); one text costs one unit.
  Over the limit the answer is `429` with `Retry-After`.
- Errors always have the form `{"error": {"code": "...", "message": "..."}}`.
- Logs record client name, counts, language and latency, never the text. Validation errors
  do not echo the input.
- Terms: research and evaluation use only (see `LICENSE-MODEL.md`).

A Gradio demo (`python demo/app.py`) is the human-facing version of the same model: a page
where you type EN/PT text and see the prediction.

## Serving

```bash
uvicorn api.main:app --reload           # http://127.0.0.1:8000  (interactive docs at /docs)
```

The container images under `deploy/` package the API and demo for identical deployment on
any host. Docker is a deployment convenience. The API runs directly from the virtual
environment without it.

## Product model selection

The served model is a Pareto trade-off, not simply the top macro-F1. Axes: quality
(macro-F1, recall-on-hate), latency (p50/p95), on-disk size, calibration (ECE), identity
bias, and license.

| Profile | Model | Why |
|---------|-------|-----|
| Best quality | XLM-R | Highest macro-F1; needs a GPU for low latency |
| **Lightweight CPU MVP** | **tfidf_logreg** | p95 1.6 ms, 3.6 MB, self-contained, within ~4 pts of XLM-R |
| Calibrated scores | sbert_lgbm | Best ECE, at the cost of a 470 MB encoder |

`hsc product` recomputes this table (and the Pareto front) whenever metrics change.

!!! danger "License gate"
    All current models are trained on the full corpus and are therefore **research-only**.
    A commercially clear model must be retrained on permissively licensed data (the
    whitelist currently holds only the Apache-2.0 source). Redistribution is bound by the
    individual dataset licenses.
