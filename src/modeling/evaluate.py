"""Evaluate baseline, ML, and SARIMAX forecasts."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.modeling.predict import GROUP_COLS, TARGET_COL, load_feature_data, load_model_bundle, predict_frame, split_time

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up module logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute bounded MAPE."""
    denom = np.where(np.abs(y_true) < 1.0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape(y_true, y_pred),
    }


def run_sarimax(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Forecast with grouped SARIMAX models."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        LOGGER.warning("statsmodels unavailable, using baseline fallback for SARIMAX")
        return test["current_week_visit_count"].to_numpy(dtype=float)

    preds = pd.Series(index=test.index, dtype=float)
    for group_key, test_group in test.groupby(GROUP_COLS):
        train_group = train[(train[GROUP_COLS[0]] == group_key[0]) & (train[GROUP_COLS[1]] == group_key[1])].sort_values("week_start")
        horizon = len(test_group)
        if train_group.empty:
            preds.loc[test_group.index] = test_group["current_week_visit_count"].to_numpy(dtype=float)
            continue

        y_train = train_group[TARGET_COL].to_numpy(dtype=float)
        fallback = float(y_train[-1])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_train,
                    order=(1, 0, 1),
                    seasonal_order=(1, 0, 0, 4),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)
                pred = np.asarray(result.forecast(steps=horizon), dtype=float)
        except Exception:
            pred = np.repeat(fallback, horizon)

        ordered_idx = test_group.sort_values("week_start").index
        preds.loc[ordered_idx] = pred

    out = preds.reindex(test.index).to_numpy(dtype=float)
    nan_mask = np.isnan(out)
    if nan_mask.any():
        out[nan_mask] = test.loc[nan_mask, "current_week_visit_count"].to_numpy(dtype=float)
    return out


def save_breakdowns(test: pd.DataFrame, columns: list[str], report_dir: Path) -> None:
    """Save city and landmark metric breakdowns."""
    report_dir.mkdir(parents=True, exist_ok=True)
    city_rows: list[dict[str, object]] = []
    lm_rows: list[dict[str, object]] = []

    for model_col in columns:
        for city, grp in test.groupby("city"):
            m = metrics(grp[TARGET_COL].to_numpy(), grp[model_col].to_numpy())
            city_rows.append({"model": model_col, "city": city, **m})
        for (city, landmark_id), grp in test.groupby(["city", "landmark_id"]):
            m = metrics(grp[TARGET_COL].to_numpy(), grp[model_col].to_numpy())
            lm_rows.append({"model": model_col, "city": city, "landmark_id": landmark_id, **m})

    pd.DataFrame(city_rows).to_csv(report_dir / "metrics_by_city.csv", index=False)
    pd.DataFrame(lm_rows).to_csv(report_dir / "metrics_by_landmark.csv", index=False)


def plot_top_landmarks(test: pd.DataFrame, figures_dir: Path) -> None:
    """Save actual-vs-predicted plots for top landmarks."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    top_landmarks = (
        test.groupby("landmark_id")[TARGET_COL]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    for landmark_id in top_landmarks:
        subset = test[test["landmark_id"] == landmark_id].sort_values("week_start")
        plt.figure(figsize=(10, 4))
        plt.plot(subset["week_start"], subset[TARGET_COL], label="actual", linewidth=2.2)
        plt.plot(subset["week_start"], subset["pred_baseline"], label="baseline", alpha=0.85)
        plt.plot(subset["week_start"], subset["pred_ml"], label="random_forest", alpha=0.85)
        plt.plot(subset["week_start"], subset["pred_sarimax"], label="sarimax", alpha=0.85)
        plt.title(f"{landmark_id} weekly demand forecast")
        plt.xlabel("week_start")
        plt.ylabel("next_week_visit_count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"actual_vs_pred_{landmark_id}.png", dpi=120)
        plt.close()


def evaluate_model(data_path: Path, model_path: Path, test_weeks: int, report_dir: Path, figures_dir: Path) -> dict[str, dict[str, float]]:
    """Run full model comparison and save artifacts."""
    frame = load_feature_data(data_path)
    train, test, cutoff = split_time(frame, test_weeks=test_weeks)
    bundle = load_model_bundle(model_path)

    test = test.sort_values(["city", "landmark_id", "week_start"]).reset_index(drop=True)
    test["pred_baseline"] = test["current_week_visit_count"].to_numpy(dtype=float)
    test["pred_ml"] = predict_frame(bundle, test)
    test["pred_sarimax"] = run_sarimax(train, test)

    summary = {
        "baseline": metrics(test[TARGET_COL].to_numpy(), test["pred_baseline"].to_numpy()),
        "random_forest": metrics(test[TARGET_COL].to_numpy(), test["pred_ml"].to_numpy()),
        "sarimax": metrics(test[TARGET_COL].to_numpy(), test["pred_sarimax"].to_numpy()),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics_summary.json").write_text(json.dumps({"cutoff": str(cutoff.date()), **summary}, indent=2))
    test.to_csv(report_dir / "predictions_test.csv", index=False)
    save_breakdowns(test, ["pred_baseline", "pred_ml", "pred_sarimax"], report_dir)
    plot_top_landmarks(test, figures_dir)

    LOGGER.info("Summary metrics %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate tourism demand models")
    parser.add_argument("--data_path", type=Path, required=True, help="Feature parquet path")
    parser.add_argument("--model_path", type=Path, required=True, help="Model bundle path")
    parser.add_argument("--test_weeks", type=int, default=12, help="Holdout weeks")
    parser.add_argument("--report_dir", type=Path, default=Path("reports"), help="Report output directory")
    parser.add_argument("--figures_dir", type=Path, default=Path("reports/figures"), help="Figure output directory")
    return parser.parse_args()


def main() -> None:
    """Run evaluate CLI."""
    setup_logging()
    args = parse_args()
    evaluate_model(args.data_path, args.model_path, args.test_weeks, args.report_dir, args.figures_dir)


if __name__ == "__main__":
    main()
