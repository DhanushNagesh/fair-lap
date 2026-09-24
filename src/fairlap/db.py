"""DuckDB connection and schema.

Raw tables mirror the source payloads one-to-one; anything derived belongs in
`transform`. All writes are idempotent: a re-run of ingestion must not change
row counts, so every raw table has a natural key and loads go through
`upsert()` rather than a bare INSERT.
"""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import pandas as pd

from fairlap.config import DUCKDB_PATH

RAW_SCHEMA: dict[str, str] = {
    # table -> natural key columns, used for idempotent upserts
    "raw_sessions": "session_key",
    "raw_laps": "session_key, driver_number, lap_number",
    "raw_position": "session_key, driver_number, date",
    "raw_intervals": "session_key, driver_number, date",
    "raw_pit": "session_key, driver_number, lap_number",
    "raw_stints": "session_key, driver_number, stint_number",
    "raw_race_control": "session_key, date, category, message",
    "raw_weather": "session_key, date",
    "raw_drivers": "session_key, driver_number",
    "raw_market_meta": "condition_id",
    "raw_market_prices": "yes_token_id, ts",
    "raw_market_trades": "transaction_hash, yes_token_id",
}


def connect(path=DUCKDB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the project database, creating the parent directory if needed."""
    raise NotImplementedError


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create every raw and derived table if absent. Safe to re-run."""
    raise NotImplementedError


def upsert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    key: Sequence[str] | None = None,
) -> int:
    """Insert rows from `df`, replacing any that collide on the natural key.

    Returns the number of rows written. Key defaults to RAW_SCHEMA[table].
    """
    raise NotImplementedError
