"""Prediction utilities for weekly demand forecasts."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TARGET_COL = "next_week_visit_count"
TIME_COL = "week_start"
GROUP_COLS = ["city", "landmark_id"]
CATEGORICAL_FEATURES = ["city", "landmark_id", "category", "indoor_outdoor", "season"]
NUMERIC_FEATURES = [
    "week_of_year",
    "month",
    "is_weekend_ratio",
    "is_holiday_week",
    "visits_last_week",
    "visits_last_2_weeks",
    "rolling_mean_4w",
    "rolling_std_4w",
    "ticket_price",
    "popularity_base_score",
    "avg_visit_time_min",
    "avg_temp",
    "rainy_days_count",
    "precipitation_sum",
    "hot_days_count",
    "events_count_week",
    "max_event_intensity",
    "current_week_visit_count",
    "returning_user_ratio",
    "avg_session_duration_sec",
    "avg_party_size",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def setup_logging() -> None:
    """Set up module logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_feature_data(data_path: Path) -> pd.DataFrame:
    """Load engineered weekly features."""
    try:
        frame = pd.read_parquet(data_path)
    except Exception:
        frame = pd.read_pickle(data_path)
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL])
    frame = frame.sort_values(GROUP_COLS + [TIME_COL]).reset_index(drop=True)
    return frame


def split_time(frame: pd.DataFrame, test_weeks: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split feature frame into train and test by time."""
    weeks = sorted(frame[TIME_COL].drop_duplicates())
    if len(weeks) <= test_weeks:
        raise ValueError(f"Not enough weekly windows for split: weeks={len(weeks)} test_weeks={test_weeks}")
    cutoff = pd.Timestamp(weeks[-test_weeks])
    train = frame[frame[TIME_COL] < cutoff].copy()
    test = frame[frame[TIME_COL] >= cutoff].copy()
    return train, test, cutoff


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load model bundle from disk."""
    with model_path.open("rb") as f:
        bundle: dict[str, Any] = pickle.load(f)
    return bundle


def predict_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Generate model predictions for a frame."""
    model = bundle["model"]
    features = bundle["feature_columns"]
    return model.predict(frame[features])


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run predictions for engineered weekly features")
    parser.add_argument("--data_path", type=Path, required=True, help="Feature parquet path")
    parser.add_argument("--model_path", type=Path, required=True, help="Trained model bundle path")
    parser.add_argument("--output_path", type=Path, default=Path("reports/predictions_cli.csv"), help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    """Run prediction CLI."""
    setup_logging()
    args = parse_args()
    frame = load_feature_data(args.data_path)
    bundle = load_model_bundle(args.model_path)
    preds = predict_frame(bundle, frame)
    out = frame[[TIME_COL, *GROUP_COLS, TARGET_COL]].copy()
    out["predicted_next_week_visit_count"] = preds
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    LOGGER.info("Saved predictions rows=%s path=%s", len(out), args.output_path)


if __name__ == "__main__":
    main()
