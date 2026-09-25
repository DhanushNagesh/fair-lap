from __future__ import annotations

import pandas as pd
import pytest

from fairlap.db import RAW_SCHEMA


def test_every_raw_table_has_a_natural_key():
    """No natural key means no idempotent upsert, which means duplicate rows."""
    for table, key in RAW_SCHEMA.items():
        assert key.strip(), f"{table} has no natural key"


def test_init_schema_is_idempotent(tmp_path):
    """make ingest runs against an existing DB as often as against a fresh one."""
    from fairlap.db import RAW_SCHEMA, connect, init_schema

    con = connect(tmp_path / "t.duckdb")
    init_schema(con)
    init_schema(con)

    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert set(RAW_SCHEMA) <= tables


def test_reingest_does_not_duplicate_rows(tmp_path):
    """A second ingest of the same payload must leave row counts identical."""
    from fairlap.db import connect, init_schema, upsert

    con = connect(tmp_path / "t.duckdb")
    init_schema(con)

    laps = pd.DataFrame(
        {
            "session_key": [9999, 9999],
            "driver_number": [1, 44],
            "lap_number": [1, 1],
            "lap_duration": [91.2, 91.8],
        }
    )
    assert upsert(con, "raw_laps", laps) == 2
    assert upsert(con, "raw_laps", laps) == 2
    assert con.execute("SELECT count(*) FROM raw_laps").fetchone()[0] == 2


def test_upsert_replaces_on_the_natural_key(tmp_path):
    """A corrected value overwrites in place rather than landing beside the old one."""
    from fairlap.db import connect, init_schema, upsert

    con = connect(tmp_path / "t.duckdb")
    init_schema(con)

    row = {"session_key": [9999], "driver_number": [1], "lap_number": [1]}
    upsert(con, "raw_laps", pd.DataFrame({**row, "lap_duration": [91.2]}))
    upsert(con, "raw_laps", pd.DataFrame({**row, "lap_duration": [90.1]}))

    assert con.execute("SELECT lap_duration FROM raw_laps").fetchall() == [(90.1,)]


def test_upsert_collapses_duplicate_keys_within_one_payload(tmp_path):
    """OpenF1 can repeat a key across pages; the load must not fan it out."""
    from fairlap.db import connect, init_schema, upsert

    con = connect(tmp_path / "t.duckdb")
    init_schema(con)

    dupes = pd.DataFrame(
        {
            "session_key": [9999, 9999],
            "driver_number": [1, 1],
            "lap_number": [1, 1],
            "lap_duration": [91.2, 90.1],
        }
    )
    assert upsert(con, "raw_laps", dupes) == 1
    assert con.execute("SELECT lap_duration FROM raw_laps").fetchall() == [(90.1,)]


def test_upsert_rejects_a_frame_missing_the_key(tmp_path):
    from fairlap.db import connect, init_schema, upsert

    con = connect(tmp_path / "t.duckdb")
    init_schema(con)

    with pytest.raises(ValueError, match="lap_number"):
        upsert(con, "raw_laps", pd.DataFrame({"session_key": [9999], "driver_number": [1]}))


def test_cache_path_is_slugified(tmp_path, monkeypatch):
    """A key with slashes must not escape the cache directory."""
    from fairlap.ingest import cache

    monkeypatch.setattr(cache, "RAW_CACHE_DIR", tmp_path)
    path = cache.cache_path("openf1", "laps/../../etc/passwd")
    assert tmp_path in path.parents
    assert path.name == "laps_.._.._etc_passwd.json"


def test_cache_roundtrip(tmp_path, monkeypatch):
    from fairlap.ingest import cache

    monkeypatch.setattr(cache, "RAW_CACHE_DIR", tmp_path)
    assert cache.read("openf1", "laps_9999") is None
    cache.write("openf1", "laps_9999", [{"lap_number": 1}])
    assert cache.read("openf1", "laps_9999") == [{"lap_number": 1}]
