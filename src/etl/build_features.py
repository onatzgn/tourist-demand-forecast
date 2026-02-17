"""Build weekly training features from SQLite."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up module logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def week_start(series: pd.Series) -> pd.Series:
    """Convert timestamps to week start dates."""
    ts = pd.to_datetime(series).dt.floor("D")
    return ts - pd.to_timedelta(ts.dt.dayofweek, unit="D")


def get_season(month: pd.Series) -> pd.Series:
    """Map month to season labels."""
    return month.map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )


def add_holiday_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Add simple holiday-week flag."""
    holiday_dates = pd.to_datetime(
        [
            "2023-01-01",
            "2023-04-23",
            "2023-07-14",
            "2023-08-15",
            "2023-12-25",
            "2024-01-01",
            "2024-04-23",
            "2024-07-14",
            "2024-08-15",
            "2024-12-25",
        ]
    )
    holidays = pd.DataFrame({"holiday_date": holiday_dates})
    holidays["week_start"] = week_start(holidays["holiday_date"])
    holiday_weeks = set(holidays["week_start"].dt.date)
    frame["is_holiday_week"] = frame["week_start"].dt.date.isin(holiday_weeks)
    return frame


def load_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    """Load source tables from SQLite."""
    with sqlite3.connect(db_path) as conn:
        visits = pd.read_sql("SELECT * FROM visits", conn, parse_dates=["visit_timestamp"])
        landmark_meta = pd.read_sql("SELECT * FROM landmark_meta", conn)
        weather_daily = pd.read_sql("SELECT * FROM weather_daily", conn, parse_dates=["date"])
        events = pd.read_sql("SELECT * FROM events", conn, parse_dates=["date"])
    return {
        "visits": visits,
        "landmark_meta": landmark_meta,
        "weather_daily": weather_daily,
        "events": events,
    }


def build_weekly_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build weekly supervised dataset."""
    visits = tables["visits"].copy()
    visits["visit_date"] = pd.to_datetime(visits["visit_timestamp"]).dt.floor("D")
    visits["week_start"] = week_start(visits["visit_timestamp"])
    visits["is_weekend"] = visits["visit_date"].dt.dayofweek >= 5

    weekly_visits = (
        visits.groupby(["city", "landmark_id", "week_start"], as_index=False)
        .agg(
            current_week_visit_count=("visit_id", "size"),
            is_weekend_ratio=("is_weekend", "mean"),
            returning_user_ratio=("is_returning_user", "mean"),
            avg_session_duration_sec=("session_duration_sec", "mean"),
            avg_party_size=("party_size", "mean"),
        )
        .sort_values(["city", "landmark_id", "week_start"])
    )

    weather = tables["weather_daily"].copy()
    weather["week_start"] = week_start(weather["date"])
    weekly_weather = weather.groupby(["city", "week_start"], as_index=False).agg(
        avg_temp=("temp_c", "mean"),
        rainy_days_count=("is_rainy", "sum"),
        precipitation_sum=("precipitation_mm", "sum"),
        hot_days_count=("is_hot_day", "sum"),
    )

    events = tables["events"].copy()
    if events.empty:
        weekly_events = pd.DataFrame(columns=["city", "week_start", "events_count_week", "max_event_intensity"])
    else:
        events["week_start"] = week_start(events["date"])
        weekly_events = events.groupby(["city", "week_start"], as_index=False).agg(
            events_count_week=("event_type", "size"),
            max_event_intensity=("event_intensity", "max"),
        )

    merged = weekly_visits.merge(tables["landmark_meta"], on=["city", "landmark_id"], how="left")
    merged = merged.merge(weekly_weather, on=["city", "week_start"], how="left")
    merged = merged.merge(weekly_events, on=["city", "week_start"], how="left")
    merged["events_count_week"] = merged["events_count_week"].fillna(0).astype(int)
    merged["max_event_intensity"] = merged["max_event_intensity"].fillna(0).astype(int)

    merged["week_of_year"] = merged["week_start"].dt.isocalendar().week.astype(int)
    merged["month"] = merged["week_start"].dt.month.astype(int)
    merged["season"] = get_season(merged["month"])
    merged = add_holiday_flag(merged)

    group_keys = ["city", "landmark_id"]
    merged = merged.sort_values(group_keys + ["week_start"]).reset_index(drop=True)

    grouped = merged.groupby(group_keys, group_keys=False)
    merged["visits_last_week"] = grouped["current_week_visit_count"].shift(1)
    merged["visits_last_2_weeks"] = grouped["current_week_visit_count"].shift(2)
    merged["rolling_mean_4w"] = grouped["current_week_visit_count"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean()
    )
    merged["rolling_std_4w"] = grouped["current_week_visit_count"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std()
    )
    merged["next_week_visit_count"] = grouped["current_week_visit_count"].shift(-1)

    merged["rolling_std_4w"] = merged["rolling_std_4w"].fillna(0.0)
    cleaned = merged.dropna(
        subset=[
            "visits_last_week",
            "visits_last_2_weeks",
            "rolling_mean_4w",
            "next_week_visit_count",
        ]
    ).copy()

    numeric_cols = [
        "visits_last_week",
        "visits_last_2_weeks",
        "rolling_mean_4w",
        "rolling_std_4w",
        "avg_temp",
        "rainy_days_count",
        "precipitation_sum",
        "hot_days_count",
        "ticket_price",
        "popularity_base_score",
        "avg_visit_time_min",
        "events_count_week",
        "max_event_intensity",
        "is_weekend_ratio",
        "current_week_visit_count",
        "returning_user_ratio",
        "avg_session_duration_sec",
        "avg_party_size",
    ]
    cleaned[numeric_cols] = cleaned[numeric_cols].astype(float)
    cleaned["next_week_visit_count"] = cleaned["next_week_visit_count"].astype(float)
    cleaned["is_holiday_week"] = cleaned["is_holiday_week"].astype(int)

    return cleaned


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build weekly features")
    parser.add_argument("--db_path", type=Path, default=Path("data/db.sqlite"), help="SQLite path")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/weekly_features.parquet"),
        help="Output parquet path",
    )
    return parser.parse_args()


def main() -> None:
    """Run feature engineering CLI."""
    setup_logging()
    args = parse_args()
    tables = load_tables(args.db_path)
    features = build_weekly_features(tables)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        features.to_parquet(args.output_path, index=False)
    except ImportError:
        features.to_pickle(args.output_path)
        LOGGER.warning("Parquet engine unavailable. Saved pickle fallback at path=%s", args.output_path)
    LOGGER.info("Saved features rows=%s path=%s", len(features), args.output_path)


if __name__ == "__main__":
    main()
