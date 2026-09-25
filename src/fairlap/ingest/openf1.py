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
from contextlib import nullcontext

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


# The free tier blocks the window from 30 min before a session to 30 min after,
# so a session is only fetched once it is comfortably outside that window.
LIVE_BLACKOUT = pd.Timedelta(minutes=30)

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


def race_window(session: pd.Series) -> tuple[int, int]:
    """(start_ts, end_ts) unix seconds for a race, from OpenF1 session bounds."""
    start = pd.Timestamp(session["date_start"])
    end = pd.Timestamp(session["date_end"])
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"session {session.get('session_key')} has no usable window")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if end <= start:
        raise ValueError(f"session {session.get('session_key')} has no usable window")
    return int(start.timestamp()), int(end.timestamp())


def _align(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Coerce a payload frame to the raw table's declared columns and types.

    OpenF1 adds fields over time (`segments_sector_1`, `headshot_url`), and
    `upsert` rejects any column the table does not have, so unknown fields are
    dropped here rather than breaking the load. Typing is explicit because an
    all-null object column will not insert into an INTEGER or BOOLEAN column.
    """
    types = db.column_types(table)
    out = pd.DataFrame(index=df.index)
    for col, sql in types.items():
        series = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index, dtype="object")
        if sql == "TIMESTAMPTZ":
            out[col] = pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
        elif sql == "BOOLEAN":
            out[col] = series.astype("boolean")
        elif sql in ("INTEGER", "BIGINT"):
            out[col] = pd.to_numeric(series, errors="coerce").astype("Int64")
        elif sql == "DOUBLE":
            out[col] = pd.to_numeric(series, errors="coerce").astype("Float64")
        else:
            out[col] = series.astype("string")
    return out


def fetch_endpoint(
    endpoint: str,
    session_key: int,
    refresh: bool = False,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """One OpenF1 endpoint for one session. Cached by (endpoint, session_key)."""
    # Checked against the known list before the request goes out: OpenF1
    # answers "no rows" and "no such endpoint" with the same 404, and
    # empty_on_404 below would swallow the second one.
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    table = f"raw_{endpoint}"
    limiter = _limiter()
    with nullcontext(client) if client else httpx.Client(timeout=30.0) as http_client:
        payload = get_json(
            http_client,
            f"{OPENF1_BASE}/{endpoint}",
            params={"session_key": session_key},
            limiter=limiter,
            source=SOURCE,
            cache_key=f"{endpoint}_{session_key}",
            refresh=refresh,
            empty_on_404=True,
        )
    frame = pd.DataFrame(payload or [])
    if frame.empty:
        return pd.DataFrame(columns=list(db.column_types(table)))
    return _align(frame, table)


def completed_sessions(sessions: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Race sessions that have finished and were not cancelled.

    Anything still inside the live blackout is skipped rather than retried:
    the free tier answers those with an error, and a cached error would be
    worse than a missing race.
    """
    if sessions.empty:
        return sessions
    now = now or pd.Timestamp.now(tz="UTC")
    end = pd.to_datetime(sessions["date_end"], utc=True, errors="coerce")
    ran = end.notna() & (end < now - LIVE_BLACKOUT)
    return sessions[ran & ~sessions["is_cancelled"].fillna(False)]


def ingest(seasons: Iterable[int], refresh: bool = False) -> dict[str, int]:
    """Load every endpoint for every race session into DuckDB.

    Returns rows written per table. Re-running with the same arguments must
    leave row counts unchanged.
    """
    sessions = fetch_sessions(seasons, refresh=refresh)
    targets = completed_sessions(sessions)

    counts: dict[str, int] = {f"raw_{endpoint}": 0 for endpoint in ENDPOINTS}
    con = db.connect()
    try:
        db.init_schema(con)
        counts["raw_sessions"] = db.upsert(con, "raw_sessions", sessions)
        with httpx.Client(timeout=30.0) as client:
            for n, (_, session) in enumerate(targets.iterrows(), start=1):
                session_key = int(session["session_key"])
                for endpoint in ENDPOINTS:
                    frame = fetch_endpoint(endpoint, session_key, refresh=refresh, client=client)
                    counts[f"raw_{endpoint}"] += db.upsert(con, f"raw_{endpoint}", frame)
                print(
                    f"  [{n}/{len(targets)}] {session['year']} "
                    f"{session['circuit_short_name']} (session {session_key})",
                    flush=True,
                )
    finally:
        con.close()
    return counts


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
