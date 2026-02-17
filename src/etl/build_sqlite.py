"""Load raw CSV files into SQLite."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up module logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_csvs(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all expected raw CSV files."""
    tables = {
        "visits": pd.read_csv(input_dir / "visits.csv", parse_dates=["visit_timestamp"]),
        "landmark_meta": pd.read_csv(input_dir / "landmark_meta.csv"),
        "weather_daily": pd.read_csv(input_dir / "weather_daily.csv", parse_dates=["date"]),
        "events": pd.read_csv(input_dir / "events.csv", parse_dates=["date"]),
    }
    return tables


def write_sqlite(tables: dict[str, pd.DataFrame], db_path: Path) -> None:
    """Write frames to SQLite tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        for name, frame in tables.items():
            frame.to_sql(name, conn, if_exists="replace", index=False)
            LOGGER.info("Wrote table=%s rows=%s", name, len(frame))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build SQLite database from raw CSV files")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw"), help="Input directory")
    parser.add_argument("--db_path", type=Path, default=Path("data/db.sqlite"), help="SQLite path")
    return parser.parse_args()


def main() -> None:
    """Run SQLite ETL CLI."""
    setup_logging()
    args = parse_args()
    tables = load_csvs(args.input_dir)
    write_sqlite(tables, args.db_path)


if __name__ == "__main__":
    main()
