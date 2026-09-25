"""The persisted feature table: rebuilds are idempotent and scoped."""

from __future__ import annotations

import pandas as pd
import pytest

from fairlap.transform.build_features import KEY, _write


def _frame(session_key: int, rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_key": [session_key] * rows,
            "lap_number": list(range(1, rows + 1)),
            "driver_number": [1] * rows,
            "position": [1.0] * rows,
        }
    )


@pytest.fixture
def con(tmp_path):
    from fairlap.db import connect

    connection = connect(tmp_path / "t.duckdb")
    try:
        yield connection
    finally:
        connection.close()


def test_a_full_build_replaces_the_table(con):
    _write(con, pd.concat([_frame(1), _frame(2)]), replace_all=True)
    _write(con, _frame(1), replace_all=True)
    assert con.execute("SELECT count(DISTINCT session_key) FROM features").fetchone()[0] == 1


def test_rebuilding_one_race_keeps_the_others(con):
    """`fairlap-features --session-key X` used to drop the table and leave X alone."""
    _write(con, pd.concat([_frame(1), _frame(2)]), replace_all=True)
    _write(con, _frame(2), replace_all=False)

    races = con.execute("SELECT count(DISTINCT session_key) FROM features").fetchone()[0]
    assert races == 2
    assert con.execute("SELECT count(*) FROM features").fetchone()[0] == 6


def test_rebuilding_the_same_race_twice_does_not_duplicate_rows(con):
    """Same rule as ingestion: a second run must not change row counts."""
    _write(con, pd.concat([_frame(1), _frame(2)]), replace_all=True)
    before = con.execute("SELECT count(*) FROM features").fetchone()[0]
    for _ in range(2):
        _write(con, _frame(2), replace_all=False)
    assert con.execute("SELECT count(*) FROM features").fetchone()[0] == before


def test_the_natural_key_is_the_grid(con):
    assert KEY == ("session_key", "lap_number", "driver_number")
