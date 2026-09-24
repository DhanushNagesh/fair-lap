"""OpenF1 ingestion: sessions, laps, position, intervals, pit, stints,
race_control, weather, drivers.

Only completed race sessions (session_name == "Race") for 2023+ are fetched --
the free tier blocks the window from 30 min before a session to 30 min after,
and live data is a paid add-on. Every endpoint is paged by session_key so a
failed season can be resumed without refetching what landed.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import pandas as pd

ENDPOINTS = (
    "laps",
    "position",
    "intervals",
    "pit",
    "stints",
    "race_control",
    "weather",
    "drivers",
)


def fetch_sessions(seasons: Iterable[int]) -> pd.DataFrame:
    """All race sessions in the given seasons, one row per session_key."""
    raise NotImplementedError


def fetch_endpoint(endpoint: str, session_key: int) -> pd.DataFrame:
    """One OpenF1 endpoint for one session. Cached by (endpoint, session_key)."""
    raise NotImplementedError


def ingest(seasons: Iterable[int], refresh: bool = False) -> dict[str, int]:
    """Load every endpoint for every race session into DuckDB.

    Returns rows written per table. Re-running with the same arguments must
    leave row counts unchanged.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OpenF1 race data into DuckDB")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--refresh", action="store_true", help="bypass the raw cache")
    args = parser.parse_args()
    print(ingest(args.seasons, refresh=args.refresh))
