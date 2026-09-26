"""Per-race raw frames, and the truncation the replay depends on.

`build_features` and `replay/stream_race` run the *same* feature functions. The
only difference between them is what they are handed: the full race, or a
`RaceInputs` truncated to everything timestamped at or before one lap end. If a
feature function reaches forward, the two disagree and
`tests/test_leakage.py::test_replay_matches_full_build` fails.

That only works if the raw frames are the single source of truth here, so
loading is centralised in this module rather than each stage running its own
SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from fairlap import db
from fairlap.config import MIN_MARKET_VOLUME_USD
from fairlap.transform.identity import resolve_market_drivers

_TIME_TABLES = {"position": "date", "intervals": "date", "race_control": "date", "market": "ts"}


@dataclass(frozen=True)
class RaceInputs:
    """Every raw frame one race needs, plus the constants known before it starts."""

    session_key: int
    year: int
    circuit_key: int
    circuit_short_name: str
    total_laps: int
    race_start: pd.Timestamp
    laps: pd.DataFrame
    position: pd.DataFrame
    intervals: pd.DataFrame
    stints: pd.DataFrame
    pit: pd.DataFrame
    race_control: pd.DataFrame
    market: pd.DataFrame

    def truncate(self, max_lap: int, cutoff: pd.Timestamp) -> RaceInputs:
        """Everything knowable at `cutoff`, for laps up to `max_lap`.

        Lap-keyed tables are cut on lap number and time-keyed tables on
        timestamp, because they are indexed differently, not because the rule
        differs. `total_laps` and `race_start` survive truncation on purpose:
        both are known before the race starts.
        """
        cut = {
            "laps": self.laps[self.laps["lap_number"] <= max_lap],
            "pit": self.pit[self.pit["lap_number"] <= max_lap],
            # lap_start is when the stint began, which is known when it begins.
            # lap_end is not loaded at all -- see `load`.
            "stints": self.stints[self.stints["lap_start"] <= max_lap],
        }
        for name, col in _TIME_TABLES.items():
            frame = getattr(self, name)
            cut[name] = frame[frame[col] <= cutoff]
        return replace(self, **cut)


def _utc(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in frame.columns:
        frame[col] = pd.to_datetime(frame[col], utc=True).astype("datetime64[ns, UTC]")
    return frame


def race_sessions(con, seasons=None) -> pd.DataFrame:
    """Race sessions that actually produced lap data, oldest first."""
    return con.execute(
        """
        SELECT s.session_key, s.year, s.circuit_key, s.circuit_short_name, s.date_start
        FROM raw_sessions s
        WHERE s.session_type = 'Race'
          AND ($1::INTEGER[] IS NULL OR s.year IN (SELECT UNNEST($1::INTEGER[])))
          AND EXISTS (SELECT 1 FROM raw_laps l WHERE l.session_key = s.session_key)
        ORDER BY s.date_start
        """,
        [list(seasons) if seasons else None],
    ).fetchdf()


def market_fills(con, session_keys=None) -> pd.DataFrame:
    """Traded minute prices per (session_key, driver_number), above the floor.

    Source is `trades`, never `history`: `/prices-history` returns a point every
    minute whether or not anyone traded, so it cannot answer whether a lap is
    scoreable. Volume uses the same effective figure Phase 0 settled on --
    Gamma reports 0 for some events whose markets demonstrably traded, so
    observed in-window notional stands in when it is larger.
    """
    meta = con.execute(
        """
        SELECT m.*,
               COALESCE(t.notional_usd, 0.0) AS notional_usd
        FROM raw_market_meta m
        LEFT JOIN (
            SELECT yes_token_id, SUM(size * price) AS notional_usd
            FROM raw_market_trades WHERE outcome = 'Yes' GROUP BY 1
        ) t USING (yes_token_id)
        WHERE m.session_key IS NOT NULL
        """
    ).fetchdf()
    drivers = con.execute("SELECT session_key, driver_number, full_name FROM raw_drivers").fetchdf()
    if meta.empty:
        return pd.DataFrame(columns=["session_key", "driver_number", "ts", "price"])

    meta = resolve_market_drivers(meta, drivers)
    meta["effective_volume_usd"] = meta[["volume_usd", "notional_usd"]].max(axis=1).fillna(0.0)
    meta = meta[meta["driver_number"].notna()]
    meta = meta[meta["effective_volume_usd"] >= MIN_MARKET_VOLUME_USD]

    prices = con.execute(
        "SELECT yes_token_id, ts, price FROM raw_market_prices WHERE source = 'trades'"
    ).fetchdf()
    out = prices.merge(
        meta[["yes_token_id", "session_key", "driver_number"]], on="yes_token_id", how="inner"
    )
    out["driver_number"] = out["driver_number"].astype("int64")
    out = _utc(out, "ts")
    if session_keys is not None:
        out = out[out["session_key"].isin(session_keys)]
    return out[["session_key", "driver_number", "ts", "price"]].sort_values("ts")


def load(session_key: int, con=None, market: pd.DataFrame | None = None) -> RaceInputs:
    """Read one race's raw frames out of DuckDB."""
    owned = con is None
    con = con or db.connect(read_only=True)
    try:
        meta = con.execute(
            "SELECT session_key, year, circuit_key, circuit_short_name, date_start "
            "FROM raw_sessions WHERE session_key = ?",
            [session_key],
        ).fetchdf()
        if meta.empty:
            raise KeyError(f"no session {session_key}")

        def read(sql: str) -> pd.DataFrame:
            return con.execute(sql, [session_key]).fetchdf()

        laps = read(
            "SELECT * FROM raw_laps WHERE session_key = ? ORDER BY driver_number, lap_number"
        )
        if laps.empty:
            raise KeyError(f"session {session_key} has no lap data")

        # lap_end is deliberately not selected. It is the stint's *final* lap,
        # which is knowable only once the stint is over, and leakage rule 2
        # rules it out. Leaving it unread means no feature can reach for it.
        stints = read(
            "SELECT session_key, driver_number, stint_number, lap_start, compound, "
            "tyre_age_at_start FROM raw_stints WHERE session_key = ? ORDER BY lap_start"
        )
        # A stint with no start lap cannot be placed on the grid and would
        # sort unpredictably in the as-of join; 29 rows across the archive.
        stints = stints[stints["lap_start"].notna()]
        # In a red-flagged race OpenF1 emits several stints sharing a lap_start
        # -- Melbourne 2025 gives driver 5 two stints beginning on lap 3 and two
        # more on lap 4. A tie leaves the as-of join to pick on frame order,
        # which differs between the full build and a truncated one, so the
        # replay disagreed with the feature table on tyre state. Keep the
        # highest-numbered stint for each start lap: that is the one the driver
        # is actually on for the rest of the lap, and one lap cannot contain
        # more than one real pit visit.
        stints = (
            stints.sort_values(["driver_number", "lap_start", "stint_number"], kind="stable")
            .drop_duplicates(["session_key", "driver_number", "lap_start"], keep="last")
            .reset_index(drop=True)
        )

        frames = {
            "position": read("SELECT * FROM raw_position WHERE session_key = ? ORDER BY date"),
            "intervals": read("SELECT * FROM raw_intervals WHERE session_key = ? ORDER BY date"),
            "race_control": read(
                "SELECT * FROM raw_race_control WHERE session_key = ? ORDER BY date"
            ),
            "pit": read("SELECT * FROM raw_pit WHERE session_key = ? ORDER BY lap_number"),
        }
    finally:
        if owned:
            con.close()

    for name, col in (("position", "date"), ("intervals", "date"), ("race_control", "date")):
        frames[name] = _utc(frames[name], col)
    laps = _utc(laps, "date_start")

    if market is None:
        market = market_fills(db.connect(read_only=True), [session_key])
    else:
        market = market[market["session_key"] == session_key]

    row = meta.iloc[0]
    return RaceInputs(
        session_key=int(row["session_key"]),
        year=int(row["year"]),
        circuit_key=int(row["circuit_key"]),
        circuit_short_name=str(row["circuit_short_name"]),
        race_start=pd.Timestamp(row["date_start"]).tz_convert("UTC"),
        # Scheduled race distance, treated as known before the race starts.
        # Derived from the laps actually run, which is the same number except
        # in a race cut short by a red flag -- see build_features for why that
        # approximation is called out rather than hidden.
        total_laps=int(laps["lap_number"].max()),
        laps=laps,
        stints=stints,
        market=market,
        **frames,
    )
