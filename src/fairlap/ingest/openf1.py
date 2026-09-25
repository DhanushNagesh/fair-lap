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

import httpx
import pandas as pd

from fairlap import db
from fairlap.config import OPENF1_BASE, OPENF1_MAX_REQ_PER_MIN, OPENF1_MAX_REQ_PER_SEC
from fairlap.ingest.http import RateLimiter, get_json

SOURCE = "openf1"

SESSION_COLUMNS = (
    "session_key",
    "meeting_key",
    "session_name",
    "session_type",
    "date_start",
    "date_end",
    "gmt_offset",
    "year",
    "country_name",
    "country_code",
    "circuit_key",
    "circuit_short_name",
    "location",
    "is_cancelled",
)

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


_LIMITER: RateLimiter | None = None


def _limiter() -> RateLimiter:
    """One limiter for the process. Rebuilding it per call hands back a full
    token bucket every time, which enforces nothing across a loop."""
    global _LIMITER
    if _LIMITER is None:
        _LIMITER = RateLimiter(OPENF1_MAX_REQ_PER_SEC, OPENF1_MAX_REQ_PER_MIN)
    return _LIMITER


def fetch_sessions(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """All race sessions in the given seasons, one row per session_key.

    `session_name == "Race"` is what excludes sprints: a sprint is a separate
    session with its own key, and treating one as a race would put two sets of
    laps under one event.
    """
    limiter = _limiter()
    frames = []
    with httpx.Client(timeout=30.0) as client:
        for season in seasons:
            payload = get_json(
                client,
                f"{OPENF1_BASE}/sessions",
                params={"year": season, "session_name": "Race"},
                limiter=limiter,
                source=SOURCE,
                cache_key=f"sessions_{season}",
                refresh=refresh,
            )
            if payload:
                frames.append(pd.DataFrame(payload))

    if not frames:
        return pd.DataFrame(columns=SESSION_COLUMNS)

    sessions = pd.concat(frames, ignore_index=True)
    for col in SESSION_COLUMNS:
        if col not in sessions.columns:
            sessions[col] = pd.NA
    for col in ("date_start", "date_end"):
        sessions[col] = pd.to_datetime(sessions[col], utc=True, format="ISO8601")
    return sessions[list(SESSION_COLUMNS)]


def ingest_sessions(seasons: Iterable[int], refresh: bool = False) -> int:
    """Load the race-session spine into raw_sessions. Phase 0 needs only this."""
    sessions = fetch_sessions(seasons, refresh=refresh)
    con = db.connect()
    try:
        db.init_schema(con)
        return db.upsert(con, "raw_sessions", sessions)
    finally:
        con.close()


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
    parser.add_argument(
        "--sessions-only",
        action="store_true",
        help="load only the race-session spine (Phase 0)",
    )
    args = parser.parse_args()
    if args.sessions_only:
        print({"raw_sessions": ingest_sessions(args.seasons, refresh=args.refresh)})
        return
    print(ingest(args.seasons, refresh=args.refresh))
