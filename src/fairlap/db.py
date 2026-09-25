"""DuckDB connection and schema.

Raw tables mirror the source payloads one-to-one; anything derived belongs in
`transform`. All writes are idempotent: a re-run of ingestion must not change
row counts, so every raw table has a natural key and loads go through
`upsert()` rather than a bare INSERT.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
    # source is part of the key: the quote series and the trade-derived series
    # are both stored, and they collide minute for minute otherwise.
    "raw_market_prices": "yes_token_id, ts, source",
    # One transaction can carry two fills (same wallet, same price, seconds
    # apart), so the hash alone is not unique within a market.
    "raw_market_trades": "transaction_hash, yes_token_id, ts",
}

# Column types are the source payload's, not ours. Two OpenF1 fields look
# numeric and are not: `gap_to_leader` and `interval` come back as "+1 LAP" for
# lapped cars, so they stay VARCHAR here and are parsed in transform/.
_RAW_DDL: dict[str, str] = {
    "raw_sessions": """
        session_key         BIGINT,
        meeting_key         BIGINT,
        session_name        VARCHAR,
        session_type        VARCHAR,
        date_start          TIMESTAMPTZ,
        date_end            TIMESTAMPTZ,
        gmt_offset          VARCHAR,
        year                INTEGER,
        country_name        VARCHAR,
        country_code        VARCHAR,
        circuit_key         BIGINT,
        circuit_short_name  VARCHAR,
        location            VARCHAR,
        is_cancelled        BOOLEAN
    """,
    "raw_laps": """
        session_key         BIGINT,
        meeting_key         BIGINT,
        driver_number       INTEGER,
        lap_number          INTEGER,
        date_start          TIMESTAMPTZ,
        lap_duration        DOUBLE,
        duration_sector_1   DOUBLE,
        duration_sector_2   DOUBLE,
        duration_sector_3   DOUBLE,
        i1_speed            INTEGER,
        i2_speed            INTEGER,
        st_speed            INTEGER,
        is_pit_out_lap      BOOLEAN
    """,
    "raw_position": """
        session_key         BIGINT,
        meeting_key         BIGINT,
        driver_number       INTEGER,
        date                TIMESTAMPTZ,
        position            INTEGER
    """,
    "raw_intervals": """
        session_key         BIGINT,
        meeting_key         BIGINT,
        driver_number       INTEGER,
        date                TIMESTAMPTZ,
        gap_to_leader       VARCHAR,
        interval            VARCHAR
    """,
    "raw_pit": """
        session_key         BIGINT,
        meeting_key         BIGINT,
        driver_number       INTEGER,
        lap_number          INTEGER,
        date                TIMESTAMPTZ,
        pit_duration        DOUBLE
    """,
    "raw_stints": """
        session_key         BIGINT,
        meeting_key         BIGINT,
        driver_number       INTEGER,
        stint_number        INTEGER,
        lap_start           INTEGER,
        lap_end             INTEGER,
        compound            VARCHAR,
        tyre_age_at_start   INTEGER
    """,
    "raw_race_control": """
        session_key         BIGINT,
        meeting_key         BIGINT,
        date                TIMESTAMPTZ,
        category            VARCHAR,
        message             VARCHAR,
        flag                VARCHAR,
        scope               VARCHAR,
        sector              INTEGER,
        driver_number       INTEGER,
        lap_number          INTEGER
    """,
    "raw_weather": """
        session_key         BIGINT,
        meeting_key         BIGINT,
        date                TIMESTAMPTZ,
        air_temperature     DOUBLE,
        track_temperature   DOUBLE,
        humidity            DOUBLE,
        pressure            DOUBLE,
        rainfall            INTEGER,
        wind_direction      INTEGER,
        wind_speed          DOUBLE
    """,
    "raw_drivers": """
        session_key         BIGINT,
        meeting_key         BIGINT,
        driver_number       INTEGER,
        full_name           VARCHAR,
        name_acronym        VARCHAR,
        team_name           VARCHAR,
        country_code        VARCHAR
    """,
    # session_key is nullable on purpose: an event we could not match to an
    # OpenF1 session is kept so scan_market_coverage can report it.
    "raw_market_meta": """
        condition_id        VARCHAR,
        event_id            VARCHAR,
        event_title         VARCHAR,
        event_slug          VARCHAR,
        market_slug         VARCHAR,
        question            VARCHAR,
        yes_token_id        VARCHAR,
        no_token_id         VARCHAR,
        driver_name         VARCHAR,
        driver_number       INTEGER,
        session_key         BIGINT,
        event_date          DATE,
        event_date_source   VARCHAR,
        match_note          VARCHAR,
        volume_usd          DOUBLE,
        closed              BOOLEAN,
        outcome_yes         BOOLEAN
    """,
    # Prices are stored with the vig. De-vigging depends on which drivers clear
    # the volume floor, which is a transform/ decision.
    "raw_market_prices": """
        yes_token_id        VARCHAR,
        ts                  TIMESTAMPTZ,
        price               DOUBLE,
        source              VARCHAR
    """,
    "raw_market_trades": """
        transaction_hash    VARCHAR,
        yes_token_id        VARCHAR,
        ts                  TIMESTAMPTZ,
        price               DOUBLE,
        size                DOUBLE,
        side                VARCHAR,
        outcome             VARCHAR
    """,
}


def connect(path=DUCKDB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the project database, creating the parent directory if needed."""
    path = Path(path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create every raw table if absent. Safe to re-run.

    Derived tables are created by the stage that owns them, so a schema change
    in transform/ never has to be mirrored here.
    """
    for table, ddl in _RAW_DDL.items():
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({ddl})")


def column_types(table: str) -> dict[str, str]:
    """Declared columns of a raw table, name -> SQL type.

    Ingestion aligns payload frames against this rather than against a second
    hand-maintained column list, so a schema change has one place to happen.
    """
    types: dict[str, str] = {}
    for line in _RAW_DDL[table].strip().splitlines():
        parts = line.strip().rstrip(",").split()
        if len(parts) >= 2:
            types[parts[0]] = parts[1]
    return types


def _key_columns(table: str, key: Sequence[str] | None) -> list[str]:
    if key is not None:
        return list(key)
    if table not in RAW_SCHEMA:
        raise KeyError(f"{table} is not in RAW_SCHEMA; pass an explicit key")
    return [c.strip() for c in RAW_SCHEMA[table].split(",")]


def upsert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    key: Sequence[str] | None = None,
) -> int:
    """Insert rows from `df`, replacing any that collide on the natural key.

    Returns the number of rows written. Key defaults to RAW_SCHEMA[table].

    DuckDB has no MERGE, so this is DELETE ... USING the staged frame followed
    by INSERT, both inside one transaction: a failed load leaves the table as
    it was rather than half-deleted.
    """
    key_cols = _key_columns(table, key)
    if df is None or df.empty:
        return 0

    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: frame is missing key column(s) {missing}")

    table_cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    unknown = [c for c in df.columns if c not in table_cols]
    if unknown:
        raise ValueError(f"{table}: frame has column(s) not in the table: {unknown}")

    # A payload can repeat a key within one page. Keep the last occurrence so
    # the DELETE + INSERT can't reintroduce the duplicate it just removed.
    staged = df.drop_duplicates(subset=key_cols, keep="last")

    cols = list(staged.columns)
    col_list = ", ".join(cols)
    on_clause = " AND ".join(f"t.{c} IS NOT DISTINCT FROM s.{c}" for c in key_cols)

    con.register("_upsert_staged", staged)
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(f"DELETE FROM {table} t USING _upsert_staged s WHERE {on_clause}")
        con.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _upsert_staged")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("_upsert_staged")

    return len(staged)
