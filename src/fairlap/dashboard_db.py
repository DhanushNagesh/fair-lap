"""Build the small read-only DuckDB file the deployed dashboard runs on.

The full database is ~50MB, gitignored, and takes a from-scratch ingest to
rebuild, so a hosted dashboard can't use it. This copies only the tables and
columns `dashboard/app.py` reads into `data/dashboard.duckdb`, which is small
enough to commit.

Before copying it replays every held-out race that has a market, so the race
tab has all of them and not just whichever races someone replayed by hand.

    uv run fairlap-dashboard-db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fairlap import db
from fairlap.config import DASHBOARD_DB_PATH, DUCKDB_PATH, TEST_SEASONS
from fairlap.replay.stream_race import _DDL, KEY, replay

EVAL_TABLES = (
    "eval_overall",
    "eval_by_phase",
    "eval_by_condition",
    "eval_by_season",
    "eval_calibration",
    "eval_retention",
)

# (table, SELECT over the attached full DB). Features are cut to the columns
# the dashboard queries and to races with a market: the coverage tab drops
# races with no priced row anyway, and the other 35 races are most of the file.
COPIES = [(t, f"SELECT * FROM src.{t}") for t in EVAL_TABLES] + [
    ("replay_predictions", "SELECT * FROM src.replay_predictions"),
    ("raw_sessions", "SELECT * FROM src.raw_sessions"),
    ("raw_drivers", "SELECT * FROM src.raw_drivers"),
    (
        "features",
        """
        SELECT session_key, lap_number, driver_number, sc_active, vsc_active,
               red_flag_active, p_market, p_market_prerace
        FROM src.features
        WHERE session_key IN (
            SELECT session_key FROM src.features WHERE p_market IS NOT NULL
        )
        """,
    ),
]


def races_to_replay(model_name: str) -> list[int]:
    con = db.connect(read_only=True)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT session_key FROM predictions p
            JOIN features f USING (session_key, lap_number, driver_number)
            WHERE p.fold = -2 AND p.model = ? AND f.p_market IS NOT NULL
              AND f.season IN (SELECT UNNEST(?::BIGINT[]))
            ORDER BY session_key
            """,
            [model_name, list(TEST_SEASONS)],
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def replay_all(model_name: str = "gbm") -> int:
    keys = races_to_replay(model_name)
    for i, key in enumerate(keys, 1):
        out = replay(key, model_name, verbose=False)
        con = db.connect()
        try:
            con.execute(f"CREATE TABLE IF NOT EXISTS replay_predictions ({_DDL})")
            db.upsert(con, "replay_predictions", out, key=KEY)
        finally:
            con.close()
        print(f"replayed {i}/{len(keys)}  session {key}  {len(out)} rows")
    return len(keys)


def build(src: Path = DUCKDB_PATH, dest: Path = DASHBOARD_DB_PATH) -> dict[str, int]:
    """Write `dest` from scratch. Returns row counts per table."""
    dest = Path(dest)
    tmp = dest.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = db.connect(tmp)
    counts = {}
    try:
        con.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
        for table, select in COPIES:
            con.execute(f"CREATE TABLE {table} AS {select}")
            counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        con.execute("DETACH src")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    # Written to a temp file and renamed so a failed build never leaves a
    # half-copied file where the dashboard will open it.
    tmp.replace(dest)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gbm")
    parser.add_argument("--no-replay", action="store_true", help="copy replay_predictions as it is")
    args = parser.parse_args()
    if not args.no_replay:
        replay_all(args.model)
    counts = build()
    for table, n in counts.items():
        print(f"{table:<20} {n:>7,}")
    size = DASHBOARD_DB_PATH.stat().st_size / 1e6
    print(f"wrote {DASHBOARD_DB_PATH} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
