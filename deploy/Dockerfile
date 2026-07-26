# CPU inference image for the FastAPI product.
# Build from repo root:  docker build -f deploy/Dockerfile -t hsc-api .
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Serving deps first (better layer caching)
COPY deploy/requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

# Package (installed without re-pulling the heavy training stack)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e . --no-deps

# App code, configs, and the trained model bundle
COPY api ./api
COPY configs ./configs
COPY models ./models

# non-root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
