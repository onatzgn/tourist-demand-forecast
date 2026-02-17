"""Project configuration values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DB_PATH = Path("data/db.sqlite")
MODEL_PATH = Path("models/model.pkl")
FIGURES_DIR = Path("reports/figures")

CITIES = ["Istanbul", "Paris", "Rome", "London", "Barcelona"]
LANDMARKS = {
    "Istanbul": [
        ("hagia_sophia", "Hagia Sophia", "religious", 28.0, "indoor", 95, 0.93),
        ("topkapi_palace", "Topkapi Palace", "historical", 22.0, "indoor", 120, 0.84),
        ("galata_tower", "Galata Tower", "viewpoint", 17.0, "outdoor", 45, 0.78),
    ],
    "Paris": [
        ("eiffel_tower", "Eiffel Tower", "viewpoint", 31.0, "outdoor", 80, 0.98),
        ("louvre", "Louvre Museum", "museum", 21.0, "indoor", 140, 0.96),
        ("notre_dame", "Notre-Dame", "religious", 11.0, "indoor", 65, 0.81),
    ],
    "Rome": [
        ("colosseum", "Colosseum", "historical", 24.0, "outdoor", 90, 0.95),
        ("vatican_museums", "Vatican Museums", "museum", 29.0, "indoor", 160, 0.92),
        ("pantheon", "Pantheon", "historical", 8.0, "indoor", 50, 0.76),
    ],
    "London": [
        ("tower_of_london", "Tower of London", "historical", 35.0, "outdoor", 85, 0.86),
        ("british_museum", "British Museum", "museum", 0.0, "indoor", 130, 0.94),
        ("westminster_abbey", "Westminster Abbey", "religious", 32.0, "indoor", 70, 0.82),
    ],
    "Barcelona": [
        ("sagrada_familia", "Sagrada Familia", "religious", 29.0, "indoor", 100, 0.97),
        ("park_guell", "Park Guell", "historical", 14.0, "outdoor", 75, 0.88),
        ("casa_batllo", "Casa Batllo", "museum", 32.0, "indoor", 60, 0.79),
    ],
}

CITY_TEMP_BASE = {
    "Istanbul": 16.0,
    "Paris": 13.5,
    "Rome": 17.5,
    "London": 11.0,
    "Barcelona": 18.0,
}

CITY_PRECIP_BASE = {
    "Istanbul": 1.9,
    "Paris": 2.2,
    "Rome": 2.0,
    "London": 2.6,
    "Barcelona": 1.5,
}

EVENT_TYPES = ["festival", "holiday", "concert", "sport"]


@dataclass(frozen=True)
class SplitConfig:
    """Time split boundaries."""

    test_weeks: int = 12


DEFAULT_SPLIT = SplitConfig()
