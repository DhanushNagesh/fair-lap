from __future__ import annotations

import pytest

from fairlap.db import RAW_SCHEMA


def test_every_raw_table_has_a_natural_key():
    """No natural key means no idempotent upsert, which means duplicate rows."""
    for table, key in RAW_SCHEMA.items():
        assert key.strip(), f"{table} has no natural key"


@pytest.mark.xfail(reason="Phase 1 not implemented", raises=NotImplementedError)
def test_init_schema_is_idempotent(tmp_path):
    from fairlap.db import connect, init_schema

    con = connect(tmp_path / "t.duckdb")
    init_schema(con)
    init_schema(con)
    raise NotImplementedError


@pytest.mark.xfail(reason="Phase 1 not implemented", raises=NotImplementedError)
def test_reingest_does_not_duplicate_rows(tmp_path):
    from fairlap.db import connect, upsert

    connect(tmp_path / "t.duckdb")
    upsert(None, "raw_laps", None)
    raise NotImplementedError


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
