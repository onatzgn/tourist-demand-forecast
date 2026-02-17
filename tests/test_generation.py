"""Tests for synthetic data generation."""

from __future__ import annotations

import pandas as pd

from src.data_generation.generate import generate_dataset


def test_generate_dataset_outputs(tmp_path) -> None:
    output_dir = tmp_path / "raw"
    paths = generate_dataset(rows=5000, output_dir=output_dir, seed=123)

    for path in paths.values():
        assert path.exists()

    visits = pd.read_csv(paths["visits"], parse_dates=["visit_timestamp"])
    weather = pd.read_csv(paths["weather_daily"], parse_dates=["date"])
    events = pd.read_csv(paths["events"], parse_dates=["date"])

    required_visit_cols = {
        "visit_id",
        "user_id",
        "city",
        "landmark_id",
        "visit_timestamp",
        "party_size",
        "device",
        "referral_source",
        "session_duration_sec",
        "is_returning_user",
        "user_segment",
    }
    assert required_visit_cols.issubset(set(visits.columns))
    assert len(visits) == 5000
    assert visits["city"].nunique() == 5
    assert visits["landmark_id"].nunique() >= 12
    assert visits["visit_timestamp"].min() >= pd.Timestamp("2023-01-01")
    assert visits["visit_timestamp"].max() <= pd.Timestamp("2024-12-31 23:59:59")

    assert weather.groupby(["city", "date"]).size().max() == 1
    assert len(events) > 0
