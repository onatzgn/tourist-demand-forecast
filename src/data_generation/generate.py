"""Generate synthetic tourism demand data."""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CITIES, CITY_PRECIP_BASE, CITY_TEMP_BASE, EVENT_TYPES, LANDMARKS

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up module logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def choose(options: list[str], probs: list[float], size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample choices from weighted options."""
    return rng.choice(options, size=size, p=np.array(probs, dtype=float))


def build_landmark_meta() -> pd.DataFrame:
    """Build static landmark metadata."""
    rows: list[dict[str, Any]] = []
    for city, landmarks in LANDMARKS.items():
        for landmark_id, landmark_name, category, ticket_price, indoor_outdoor, avg_visit_time_min, popularity in landmarks:
            rows.append(
                {
                    "landmark_id": landmark_id,
                    "landmark_name": landmark_name,
                    "city": city,
                    "category": category,
                    "ticket_price": float(ticket_price),
                    "indoor_outdoor": indoor_outdoor,
                    "avg_visit_time_min": int(avg_visit_time_min),
                    "popularity_base_score": float(popularity),
                }
            )
    return pd.DataFrame(rows)


def generate_weather(cities: list[str], dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Generate daily weather by city."""
    frames: list[pd.DataFrame] = []
    day_of_year = dates.dayofyear.to_numpy()
    season_curve = np.sin(2.0 * np.pi * (day_of_year - 170) / 365.25)
    rain_curve = 1.0 + 0.35 * np.cos(2.0 * np.pi * (day_of_year - 20) / 365.25)
    for city in cities:
        temp_base = CITY_TEMP_BASE[city]
        precip_base = CITY_PRECIP_BASE[city]
        temp = temp_base + 10.0 * season_curve + rng.normal(0.0, 3.8, size=len(dates))
        precip = rng.gamma(shape=1.6, scale=precip_base, size=len(dates)) * rain_curve
        frame = pd.DataFrame(
            {
                "city": city,
                "date": dates,
                "temp_c": np.round(temp, 2),
                "precipitation_mm": np.round(precip, 2),
            }
        )
        frame["is_rainy"] = frame["precipitation_mm"] > 3.0
        frame["is_hot_day"] = frame["temp_c"] >= 28.0
        frame["is_cold_day"] = frame["temp_c"] <= 8.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def generate_events(cities: list[str], dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Generate sparse city-level events."""
    rows: list[dict[str, Any]] = []
    for city in cities:
        for date in dates:
            weekend_boost = 0.02 if date.dayofweek >= 5 else 0.0
            summer_boost = 0.02 if date.month in {6, 7, 8, 9} else 0.0
            if rng.random() < (0.04 + weekend_boost + summer_boost):
                rows.append(
                    {
                        "city": city,
                        "date": date,
                        "event_type": rng.choice(EVENT_TYPES, p=[0.28, 0.22, 0.30, 0.20]),
                        "event_intensity": int(rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10])),
                    }
                )
    return pd.DataFrame(rows)


def build_daily_demand(
    landmark_meta: pd.DataFrame,
    weather_daily: pd.DataFrame,
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build daily demand weights by landmark."""
    calendar = pd.DataFrame({"date": dates})
    calendar["month"] = calendar["date"].dt.month
    calendar["day_of_week"] = calendar["date"].dt.dayofweek
    calendar["is_weekend"] = (calendar["day_of_week"] >= 5).astype(int)

    demand = landmark_meta.merge(calendar, how="cross")
    demand = demand.merge(weather_daily, on=["city", "date"], how="left")
    if events.empty:
        demand["event_intensity"] = 0
        demand["event_type"] = "none"
    else:
        demand = demand.merge(events, on=["city", "date"], how="left")
        demand["event_intensity"] = demand["event_intensity"].fillna(0).astype(int)
        demand["event_type"] = demand["event_type"].fillna("none")

    season_factor = np.select(
        [
            demand["month"].isin([6, 7, 8]),
            demand["month"].isin([4, 5, 9, 10]),
            demand["month"].isin([11, 12]),
        ],
        [1.35, 1.15, 0.95],
        default=0.88,
    )
    weekend_factor = np.where(demand["is_weekend"].eq(1), 1.20, 1.0)
    event_factor = 1.0 + demand["event_intensity"] * 0.12
    rain_factor = np.where(
        demand["indoor_outdoor"].eq("outdoor") & demand["is_rainy"],
        0.72,
        np.where(demand["indoor_outdoor"].eq("indoor") & demand["is_rainy"], 1.04, 1.0),
    )
    temp_factor = np.where(
        demand["indoor_outdoor"].eq("outdoor"),
        np.where(demand["temp_c"] > 31.0, 0.88, np.where(demand["temp_c"] < 6.0, 0.84, 1.04)),
        1.0,
    )
    city_factor = demand["city"].map({"Paris": 1.12, "Rome": 1.08, "Barcelona": 1.04, "Istanbul": 1.0, "London": 1.02})
    ticket_penalty = np.exp(-demand["ticket_price"] / 120.0)

    base = 32.0 + demand["popularity_base_score"] * 140.0
    demand["demand_score"] = (
        base
        * season_factor
        * weekend_factor
        * event_factor
        * rain_factor
        * temp_factor
        * city_factor
        * ticket_penalty
    )
    demand["demand_score"] = demand["demand_score"].clip(lower=0.2)
    return demand


def sample_visits(rows: int, demand: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample synthetic visits from demand weights."""
    weights = demand["demand_score"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    sampled_idx = rng.choice(demand.index.to_numpy(), size=rows, p=weights)
    sampled = demand.loc[sampled_idx, ["city", "landmark_id", "date", "avg_visit_time_min"]].reset_index(drop=True)

    user_pool = max(30_000, rows // 7)
    user_ids = np.arange(1, user_pool + 1)
    ranks = np.arange(1, user_pool + 1)
    user_weights = (1.0 / np.power(ranks, 1.08)).astype(float)
    user_weights /= user_weights.sum()

    sampled_users = rng.choice(user_ids, size=rows, p=user_weights)
    profiles = pd.DataFrame({"user_id": user_ids})
    profiles["user_segment"] = choose(["budget", "comfort", "luxury"], [0.5, 0.35, 0.15], user_pool, rng)

    device_pref = {
        "budget": (["android", "ios", "web"], [0.62, 0.22, 0.16]),
        "comfort": (["android", "ios", "web"], [0.41, 0.42, 0.17]),
        "luxury": (["android", "ios", "web"], [0.20, 0.66, 0.14]),
    }
    segment_map = profiles.set_index("user_id")["user_segment"]
    sampled["user_id"] = sampled_users
    sampled["user_segment"] = sampled["user_id"].map(segment_map)

    sampled["device"] = "web"
    for segment, (options, probs) in device_pref.items():
        mask = sampled["user_segment"].eq(segment)
        sampled.loc[mask, "device"] = choose(options, probs, int(mask.sum()), rng)

    party_size_probs = {
        "budget": [0.46, 0.30, 0.13, 0.07, 0.03, 0.01],
        "comfort": [0.30, 0.32, 0.19, 0.11, 0.06, 0.02],
        "luxury": [0.25, 0.31, 0.20, 0.12, 0.08, 0.04],
    }
    sampled["party_size"] = 1
    for segment, probs in party_size_probs.items():
        mask = sampled["user_segment"].eq(segment)
        sampled.loc[mask, "party_size"] = choose(["1", "2", "3", "4", "5", "6"], probs, int(mask.sum()), rng).astype(int)

    sampled["visit_timestamp"] = sampled["date"] + pd.to_timedelta(rng.integers(8 * 3600, 21 * 3600, size=rows), unit="s")
    sampled = sampled.sort_values(["user_id", "visit_timestamp"]).reset_index(drop=True)
    sampled["is_returning_user"] = sampled.groupby("user_id").cumcount().gt(0)

    sampled["referral_source"] = "organic"
    returning_mask = sampled["is_returning_user"]
    sampled.loc[returning_mask, "referral_source"] = choose(
        ["organic", "social", "ads", "influencer", "referral"],
        [0.45, 0.12, 0.08, 0.05, 0.30],
        int(returning_mask.sum()),
        rng,
    )
    sampled.loc[~returning_mask, "referral_source"] = choose(
        ["organic", "social", "ads", "influencer", "referral"],
        [0.24, 0.24, 0.31, 0.17, 0.04],
        int((~returning_mask).sum()),
        rng,
    )

    duration_noise = rng.normal(0.0, 14.0 * 60.0, size=rows)
    sampled["session_duration_sec"] = (
        sampled["avg_visit_time_min"] * 60.0 + sampled["party_size"] * 95.0 + duration_noise
    ).clip(300.0, 5.0 * 3600.0)

    sampled["visit_id"] = [str(uuid.uuid4()) for _ in range(rows)]
    out = sampled[
        [
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
        ]
    ].copy()
    out["session_duration_sec"] = out["session_duration_sec"].round().astype(int)
    out["visit_timestamp"] = pd.to_datetime(out["visit_timestamp"])
    return out.sort_values("visit_timestamp").reset_index(drop=True)


def generate_dataset(rows: int, output_dir: Path, seed: int = 42) -> dict[str, Path]:
    """Generate all raw CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

    landmark_meta = build_landmark_meta()
    weather_daily = generate_weather(CITIES, dates, rng)
    events = generate_events(CITIES, dates, rng)
    demand = build_daily_demand(landmark_meta, weather_daily, events, dates)
    visits = sample_visits(rows, demand, rng)

    output_paths = {
        "visits": output_dir / "visits.csv",
        "landmark_meta": output_dir / "landmark_meta.csv",
        "weather_daily": output_dir / "weather_daily.csv",
        "events": output_dir / "events.csv",
    }

    visits.to_csv(output_paths["visits"], index=False)
    landmark_meta.to_csv(output_paths["landmark_meta"], index=False)
    weather_daily.to_csv(output_paths["weather_daily"], index=False)
    events.to_csv(output_paths["events"], index=False)

    LOGGER.info("Generated visits=%s weather=%s events=%s", len(visits), len(weather_daily), len(events))
    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic tourism demand data")
    parser.add_argument("--rows", type=int, default=500_000, help="Number of visit rows")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Run the dataset generator CLI."""
    setup_logging()
    args = parse_args()
    generate_dataset(rows=args.rows, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
