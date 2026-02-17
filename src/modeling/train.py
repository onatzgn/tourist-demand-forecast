"""Train a production-style demand model."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.modeling.predict import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    load_feature_data,
    split_time,
)

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up module logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def make_ohe() -> OneHotEncoder:
    """Build a version-safe one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(random_state: int) -> Pipeline:
    """Build model training pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute bounded MAPE."""
    denom = np.where(np.abs(y_true) < 1.0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def train_and_save(data_path: Path, model_out: Path, test_weeks: int, random_state: int) -> dict[str, Any]:
    """Fit model and save training bundle."""
    frame = load_feature_data(data_path)
    train, test, cutoff = split_time(frame, test_weeks=test_weeks)

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(train[MODEL_FEATURES], train[TARGET_COL])

    test_pred = pipeline.predict(test[MODEL_FEATURES])
    metrics = {
        "mae": float(mean_absolute_error(test[TARGET_COL], test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(test[TARGET_COL], test_pred))),
        "mape": mape(test[TARGET_COL].to_numpy(), test_pred),
    }

    bundle = {
        "model": pipeline,
        "feature_columns": MODEL_FEATURES,
        "target_column": TARGET_COL,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split_cutoff": str(cutoff.date()),
        "quick_test_metrics": metrics,
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(bundle, f)

    LOGGER.info("Saved model path=%s train_rows=%s test_rows=%s", model_out, len(train), len(test))
    LOGGER.info("Quick holdout metrics %s", metrics)
    return bundle


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train tourism demand model")
    parser.add_argument("--data_path", type=Path, required=True, help="Feature parquet path")
    parser.add_argument("--model_out", type=Path, default=Path("models/model.pkl"), help="Model output path")
    parser.add_argument("--test_weeks", type=int, default=12, help="Holdout weeks")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    return parser.parse_args()


def main() -> None:
    """Run model training CLI."""
    setup_logging()
    args = parse_args()
    train_and_save(args.data_path, args.model_out, args.test_weeks, args.random_state)


if __name__ == "__main__":
    main()
