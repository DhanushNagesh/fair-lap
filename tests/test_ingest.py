"""Phase 1 ingestion: payload alignment, live blackout, idempotent re-runs."""

from __future__ import annotations

import pandas as pd
import pytest

from fairlap import db
from fairlap.ingest import openf1, polymarket


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point every db.connect() in the process at a throwaway file."""
    path = tmp_path / "t.duckdb"
    real_connect = db.connect
    monkeypatch.setattr(db, "connect", lambda *a, **kw: real_connect(path, **kw))
    return path


def counts(path) -> dict[str, int]:
    con = db.connect(path)
    try:
        return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in db.RAW_SCHEMA}
    finally:
        con.close()


LAP_PAYLOAD = [
    {
        "session_key": 9999,
        "meeting_key": 1,
        "driver_number": 1,
        "lap_number": 1,
        "date_start": "2025-09-21T11:00:00+00:00",
        "lap_duration": 91.2,
        "is_pit_out_lap": False,
        "segments_sector_1": [2049, 2049],
    }
]


def test_align_drops_fields_the_table_does_not_have():
    """OpenF1 adds fields over time; upsert rejects unknown columns."""
    out = openf1._align(pd.DataFrame(LAP_PAYLOAD), "raw_laps")
    assert "segments_sector_1" not in out.columns
    assert list(out.columns) == list(db.column_types("raw_laps"))


def test_align_types_absent_columns_so_they_still_insert(tmp_db):
    """A column the payload never sent must land as NULL, not break the load."""
    frame = openf1._align(pd.DataFrame(LAP_PAYLOAD), "raw_laps")
    assert frame["i1_speed"].isna().all()

    con = db.connect(tmp_db)
    db.init_schema(con)
    assert db.upsert(con, "raw_laps", frame) == 1
    assert con.execute("SELECT i1_speed FROM raw_laps").fetchone() == (None,)
    con.close()


def test_align_keeps_lapped_car_gap_strings_as_text():
    """`gap_to_leader` is "+1 LAP" for a lapped car; coercing it to a number
    would silently null out exactly the cars the model cares about."""
    payload = [
        {
            "session_key": 1,
            "driver_number": 1,
            "date": "2025-09-21T11:00:00+00:00",
            "gap_to_leader": "+1 LAP",
            "interval": "0.512",
        }
    ]
    out = openf1._align(pd.DataFrame(payload), "raw_intervals")
    assert out["gap_to_leader"].tolist() == ["+1 LAP"]


def test_empty_payload_gives_a_typed_empty_frame(monkeypatch):
    monkeypatch.setattr(openf1, "get_json", lambda *a, **kw: [])
    out = openf1.fetch_endpoint("laps", 9999, client=object())
    assert out.empty
    assert list(out.columns) == list(db.column_types("raw_laps"))


def test_unknown_endpoint_is_rejected():
    with pytest.raises(ValueError, match="car_data"):
        openf1.fetch_endpoint("car_data", 9999)


def sessions_frame(**over) -> pd.DataFrame:
    base = {
        "session_key": [9999],
        "year": [2025],
        "circuit_short_name": ["Monza"],
        "date_start": pd.to_datetime(["2025-09-07T13:00:00Z"]),
        "date_end": pd.to_datetime(["2025-09-07T15:00:00Z"]),
        "is_cancelled": [False],
    }
    return pd.DataFrame({**base, **over})


def test_a_session_inside_the_live_blackout_is_never_fetched():
    """Free OpenF1 blocks 30 min either side of a session, so a race that just
    ended must be skipped rather than retried into a cached error."""
    sessions = sessions_frame()
    just_ended = pd.Timestamp("2025-09-07T15:10:00Z")
    assert openf1.completed_sessions(sessions, now=just_ended).empty
    later = pd.Timestamp("2025-09-07T16:00:00Z")
    assert len(openf1.completed_sessions(sessions, now=later)) == 1


def test_cancelled_and_future_sessions_are_skipped():
    now = pd.Timestamp("2025-09-08T00:00:00Z")
    assert openf1.completed_sessions(sessions_frame(is_cancelled=[True]), now=now).empty
    future = sessions_frame(date_end=pd.to_datetime(["2026-09-07T15:00:00Z"]))
    assert openf1.completed_sessions(future, now=now).empty


def test_openf1_ingest_is_idempotent(tmp_db, monkeypatch):
    """The exit criterion for Phase 1: make ingest twice, row counts identical."""
    monkeypatch.setattr(openf1, "fetch_sessions", lambda *a, **kw: sessions_frame())
    monkeypatch.setattr(
        openf1,
        "fetch_endpoint",
        lambda endpoint, key, **kw: (
            openf1._align(pd.DataFrame(LAP_PAYLOAD), "raw_laps")
            if endpoint == "laps"
            else pd.DataFrame(columns=list(db.column_types(f"raw_{endpoint}")))
        ),
    )

    first = openf1.ingest([2025], refresh=False)
    after_first = counts(tmp_db)
    second = openf1.ingest([2025], refresh=False)

    assert first == second
    assert counts(tmp_db) == after_first
    assert after_first["raw_laps"] == 1
    assert after_first["raw_sessions"] == 1


def trades_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_hash": ["0xaa", "0xaa"],
            "token_id": ["yes-token", "yes-token"],
            "ts": pd.to_datetime(["2025-09-21T11:00:01Z", "2025-09-21T11:00:03Z"]),
            "price": [0.60, 0.61],
            "size": [100.0, 50.0],
            "side": ["BUY", "BUY"],
            "outcome": ["Yes", "Yes"],
        }
    )


def test_two_fills_in_one_transaction_are_both_kept(tmp_db):
    """One wallet can fill twice in a single transaction seconds apart. Keying
    on the hash alone would drop the second fill and understate volume."""
    con = db.connect(tmp_db)
    db.init_schema(con)
    fills = trades_frame().assign(yes_token_id="yes-token")
    assert db.upsert(con, "raw_market_trades", fills[list(polymarket.TRADE_TABLE_COLUMNS)]) == 2
    con.close()


def test_price_series_tags_both_sources(monkeypatch):
    """History and trades are stored side by side, not one chosen here."""
    history = pd.DataFrame({"ts": pd.to_datetime(["2025-09-21T11:00:00Z"]), "price": [0.58]})
    monkeypatch.setattr(polymarket, "fetch_prices_history", lambda *a, **kw: history)
    monkeypatch.setattr(polymarket, "fetch_trades", lambda *a, **kw: trades_frame())

    prices, trades = polymarket.price_series("yes-token", "0xcond", 0, 600)
    assert set(prices["source"]) == {"history", "trades"}
    assert len(trades) == 2


def test_price_series_falls_back_to_trades_when_history_is_empty(monkeypatch):
    """prices-history returns empty for some resolved markets (issue #216)."""
    monkeypatch.setattr(
        polymarket, "fetch_prices_history", lambda *a, **kw: pd.DataFrame(columns=["ts", "price"])
    )
    monkeypatch.setattr(polymarket, "fetch_trades", lambda *a, **kw: trades_frame())

    prices, _ = polymarket.price_series("yes-token", "0xcond", 0, 600)
    assert set(prices["source"]) == {"trades"}
    assert len(prices) == 1


def test_a_market_with_no_prices_at_all_is_not_an_error(monkeypatch):
    empty_trades = pd.DataFrame(columns=list(polymarket.TRADE_COLUMNS))
    monkeypatch.setattr(
        polymarket, "fetch_prices_history", lambda *a, **kw: pd.DataFrame(columns=["ts", "price"])
    )
    monkeypatch.setattr(polymarket, "fetch_trades", lambda *a, **kw: empty_trades)

    prices, trades = polymarket.price_series("yes-token", "0xcond", 0, 600)
    assert prices.empty and trades.empty
