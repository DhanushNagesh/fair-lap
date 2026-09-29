"""The committed dashboard snapshot has to be enough to run the whole dashboard.

These run against data/dashboard.duckdb, which is tracked, so they work on a
clean clone with no full database.
"""

from __future__ import annotations

import duckdb
import pytest

from fairlap import config
from fairlap.dashboard_db import COPIES, build

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

needs_snapshot = pytest.mark.skipif(
    not config.DASHBOARD_DB_PATH.exists(), reason="run `make dashboard-db` first"
)


@needs_snapshot
def test_every_tab_renders_from_the_snapshot(monkeypatch, tmp_path):
    # Hide the full database so the app has to fall back to the snapshot.
    monkeypatch.setattr(config, "DUCKDB_PATH", tmp_path / "missing.duckdb")
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60).run()
    assert not at.exception
    assert any("committed snapshot" in c.value for c in at.sidebar.caption)
    assert len(at.tabs) == 4
    assert not at.info, [i.value for i in at.info]
    races = at.selectbox[0].options
    assert len(races) >= 36


@needs_snapshot
def test_snapshot_has_exactly_the_copied_tables():
    con = duckdb.connect(str(config.DASHBOARD_DB_PATH), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert tables == {t for t, _ in COPIES}


def test_build_trims_features_to_races_with_a_market(tmp_path):
    src = tmp_path / "full.duckdb"
    con = duckdb.connect(str(src))
    for table, _ in COPIES:
        if table != "features":
            con.execute(f"CREATE TABLE {table} (x INTEGER)")
    con.execute(
        """
        CREATE TABLE features AS SELECT * FROM (VALUES
            (1, 1, 4, false, false, false, 0.6, 0.5, 1.23),
            (1, 2, 4, true,  false, false, NULL, 0.5, 1.24),
            (2, 1, 4, false, false, false, NULL, NULL, 1.30)
        ) t(session_key, lap_number, driver_number, sc_active, vsc_active,
            red_flag_active, p_market, p_market_prerace, pace_roll_s)
        """
    )
    con.close()

    counts = build(src, tmp_path / "small.duckdb")

    assert counts["features"] == 2
    con = duckdb.connect(str(tmp_path / "small.duckdb"), read_only=True)
    cols = [r[0] for r in con.execute("DESCRIBE features").fetchall()]
    keys = con.execute("SELECT DISTINCT session_key FROM features").fetchall()
    con.close()
    assert "pace_roll_s" not in cols
    assert keys == [(1,)]
    assert not (tmp_path / "small.tmp").exists()
