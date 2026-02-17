"""Tests for weekly feature engineering."""

from __future__ import annotations

import pandas as pd

from src.data_generation.generate import generate_dataset
from src.etl.build_features import build_weekly_features, load_tables
from src.etl.build_sqlite import load_csvs, write_sqlite


def test_build_weekly_features(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "db.sqlite"

    generate_dataset(rows=20000, output_dir=raw_dir, seed=55)
    tables = load_csvs(raw_dir)
    write_sqlite(tables, db_path)

    loaded = load_tables(db_path)
    features = build_weekly_features(loaded)

    required_cols = {
        "city",
        "landmark_id",
        "week_start",
        "week_of_year",
        "month",
        "season",
        "is_weekend_ratio",
        "is_holiday_week",
        "visits_last_week",
        "visits_last_2_weeks",
        "rolling_mean_4w",
        "rolling_std_4w",
        "ticket_price",
        "category",
        "indoor_outdoor",
        "popularity_base_score",
        "avg_temp",
        "rainy_days_count",
        "precipitation_sum",
        "hot_days_count",
        "events_count_week",
        "max_event_intensity",
        "next_week_visit_count",
    }
    assert required_cols.issubset(set(features.columns))
    assert not features["next_week_visit_count"].isna().any()
    assert not features["visits_last_week"].isna().any()
    assert (features["rolling_std_4w"] >= 0).all()

    sample = features.sort_values(["city", "landmark_id", "week_start"]).copy()
    sample["expected_last_week"] = sample.groupby(["city", "landmark_id"])["current_week_visit_count"].shift(1)
    merged = sample.dropna(subset=["expected_last_week"])
    assert (merged["visits_last_week"].round(6) == merged["expected_last_week"].round(6)).all()
    assert pd.api.types.is_datetime64_any_dtype(features["week_start"])
