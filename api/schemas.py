"""Request/response contracts for the product API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field("auto", pattern="^(auto|en|pt)$")


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)


class LanguageInfo(BaseModel):
    detected: str
    confidence: float


class Prediction(BaseModel):
    text: str
    label: str
    score: float
    language: LanguageInfo
    model_version: str


class PredictResponse(Prediction):
    latency_ms: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
