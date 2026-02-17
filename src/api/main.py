"""Minimal FastAPI service for demand forecasting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.modeling.predict import load_feature_data, load_model_bundle, predict_frame

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_PATH = Path("data/processed/weekly_features.parquet")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("reports/metrics_summary.json")

app = FastAPI(title="tourism-demand-forecast", version="0.1.0")

MODEL_BUNDLE: dict | None = None
FEATURE_DATA: pd.DataFrame | None = None


class PredictRequest(BaseModel):
    """Input payload for demand prediction."""

    city: str
    landmark_id: str
    week_start_date: str


@app.on_event("startup")
def startup() -> None:
    """Load model artifacts at startup."""
    global MODEL_BUNDLE, FEATURE_DATA
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        LOGGER.warning("Model or feature data missing. API is partially available.")
        return
    MODEL_BUNDLE = load_model_bundle(MODEL_PATH)
    FEATURE_DATA = load_feature_data(DATA_PATH)
    LOGGER.info("Loaded model and feature data")


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    """Return latest evaluation metrics."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    if MODEL_BUNDLE is not None:
        return {"quick_test_metrics": MODEL_BUNDLE.get("quick_test_metrics", {})}
    return {"message": "Metrics unavailable"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, float | str]:
    """Predict next week demand for a city-landmark-week."""
    if MODEL_BUNDLE is None or FEATURE_DATA is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")

    week_start = pd.to_datetime(payload.week_start_date)
    row = FEATURE_DATA[
        (FEATURE_DATA["city"] == payload.city)
        & (FEATURE_DATA["landmark_id"] == payload.landmark_id)
        & (FEATURE_DATA["week_start"] == week_start)
    ]
    if row.empty:
        raise HTTPException(status_code=404, detail="No matching feature row found for input")

    prediction = float(predict_frame(MODEL_BUNDLE, row.iloc[[0]])[0])
    return {
        "city": payload.city,
        "landmark_id": payload.landmark_id,
        "week_start_date": str(week_start.date()),
        "predicted_next_week_visit_count": round(prediction, 3),
    }
