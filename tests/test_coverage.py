"""Coverage scan: race windows, trade->price reconstruction, race status buckets."""

from __future__ import annotations

import pandas as pd
import pytest

from fairlap.ingest import polymarket
from fairlap.ingest import scan_market_coverage as sc


def test_race_window_is_the_session_bounds_in_unix_seconds():
    session = pd.Series(
        {
            "session_key": 9839,
            "date_start": pd.Timestamp("2025-12-07T13:00:00Z"),
            "date_end": pd.Timestamp("2025-12-07T15:00:00Z"),
        }
    )
    start, end = sc.race_window(session)
    assert (end - start) // 60 == 120
    assert pd.Timestamp(start, unit="s", tz="UTC").isoformat() == "2025-12-07T13:00:00+00:00"


def test_race_window_rejects_a_session_with_no_end():
    session = pd.Series(
        {
            "session_key": 1,
            "date_start": pd.Timestamp("2025-12-07T13:00:00Z"),
            "date_end": pd.NaT,
        }
    )
    with pytest.raises(ValueError):
        sc.race_window(session)


def trades_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_no_fills_are_converted_to_yes_prices():
    """A No fill at 0.8 is a Yes price of 0.2. Dropping them loses half the prints."""
    trades = trades_frame(
        [
            {"ts": "2025-09-21T11:00:10Z", "price": 0.80, "size": 100, "outcome": "No"},
        ]
    )
    out = polymarket.prices_from_trades(trades)
    assert out["price"].tolist() == pytest.approx([0.20])


def test_a_minute_with_no_trade_produces_no_row():
    """Never forward-fill: a minute nobody traded must be visibly absent."""
    trades = trades_frame(
        [
            {"ts": "2025-09-21T11:00:10Z", "price": 0.60, "size": 10, "outcome": "Yes"},
            {"ts": "2025-09-21T11:03:10Z", "price": 0.70, "size": 10, "outcome": "Yes"},
        ]
    )
    out = polymarket.prices_from_trades(trades)
    assert len(out) == 2
    minutes = out["ts"].dt.minute.tolist()
    assert minutes == [0, 3]


def test_prices_within_a_minute_are_size_weighted():
    trades = trades_frame(
        [
            {"ts": "2025-09-21T11:00:05Z", "price": 0.50, "size": 1, "outcome": "Yes"},
            {"ts": "2025-09-21T11:00:45Z", "price": 0.60, "size": 9, "outcome": "Yes"},
        ]
    )
    out = polymarket.prices_from_trades(trades)
    assert out["price"].tolist() == pytest.approx([0.59])
    assert out["size"].tolist() == pytest.approx([10.0])


def test_mixed_yes_and_no_fills_land_in_one_series():
    trades = trades_frame(
        [
            {"ts": "2025-09-21T11:00:05Z", "price": 0.60, "size": 10, "outcome": "Yes"},
            {"ts": "2025-09-21T11:00:45Z", "price": 0.40, "size": 10, "outcome": "No"},
        ]
    )
    out = polymarket.prices_from_trades(trades)
    assert out["price"].tolist() == pytest.approx([0.60])


def test_empty_trades_give_an_empty_series_not_an_error():
    out = polymarket.prices_from_trades(pd.DataFrame(columns=["ts", "price", "size", "outcome"]))
    assert out.empty


def races_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True)
    return df


NOW = pd.Timestamp("2026-09-24T00:00:00Z")


def race_row(key, date, markets, cancelled=False):
    base = {
        "session_key": key,
        "year": int(date[:4]),
        "circuit_short_name": f"C{key}",
        "date_start": date,
        "is_cancelled": cancelled,
    }
    if markets == 0:
        return [{**base, "condition_id": None}]
    return [{**base, "condition_id": f"0x{key}{i}"} for i in range(markets)]


def test_a_race_in_the_era_with_no_market_is_a_matcher_failure():
    """The bucket that exists so a dropped event cannot pass as an uncovered race."""
    races = races_frame(race_row(1, "2025-06-01T13:00:00Z", 0))
    status = sc.race_status(races, now=NOW)
    assert status["status"].tolist() == ["market_missing"]


def test_a_race_before_the_era_with_no_market_is_expected():
    races = races_frame(race_row(1, "2024-07-21T13:00:00Z", 0))
    assert sc.race_status(races, now=NOW)["status"].tolist() == ["no_market_expected"]


def test_a_race_before_the_era_that_does_have_a_market_is_still_scanned():
    """The 2024 British GP is an isolated early market. It is real data."""
    races = races_frame(race_row(1, "2024-07-07T13:00:00Z", 8))
    assert sc.race_status(races, now=NOW)["status"].tolist() == ["market_found"]


def test_cancelled_and_future_races_are_never_scanned():
    races = races_frame(
        race_row(1, "2026-04-12T13:00:00Z", 21, cancelled=True)
        + race_row(2, "2026-10-11T13:00:00Z", 23)
    )
    assert set(sc.race_status(races, now=NOW)["status"]) == {"not_run"}


def market_row(session_key, driver, fresh, total=120, rank=1, volume=50_000.0):
    return {
        "session_key": session_key,
        "year": 2025,
        "circuit_short_name": "Monza",
        "race_date": "2025-09-07",
        "driver_name": driver,
        "volume_usd": volume,
        "above_volume_floor": volume >= 5_000,
        "total_minutes": total,
        "quote_minutes": total,
        "trade_minutes": int(total * 0.4),
        "fresh_minutes": fresh,
        "quote_coverage": 1.0,
        "trade_coverage": 0.4,
        "fresh_coverage": fresh / total,
        "trades_in_window": 500,
        "volume_rank": rank,
        "top_n": rank <= 6,
    }


def test_summarise_keeps_every_race_and_reports_retention_instead_of_a_gate():
    """Option E: no race passes or fails. Retention says how much of it survives."""
    thin = [market_row(1, f"D{i}", fresh=24, rank=i + 1) for i in range(6)]
    dense = [market_row(2, f"D{i}", fresh=114, rank=i + 1) for i in range(6)]
    out = sc.summarise(pd.DataFrame(thin + dense))

    assert set(out["session_key"]) == {1, 2}, "a thin race is still in the eval set"
    assert "in_eval_fresh" not in out.columns, "the race-level gate is gone"
    assert out.set_index("session_key")["retention"].round(2).to_dict() == {1: 0.20, 2: 0.95}


def test_dense_coverage_is_a_subgroup_flag_not_an_eval_gate():
    thin = [market_row(1, f"D{i}", fresh=24, rank=i + 1) for i in range(6)]
    dense = [market_row(2, f"D{i}", fresh=114, rank=i + 1) for i in range(6)]
    out = sc.summarise(pd.DataFrame(thin + dense)).set_index("session_key")
    assert out["dense_coverage"].to_dict() == {1: False, 2: True}


def test_retention_ignores_drivers_outside_the_top_n():
    """A 20th-place driver's dead market must not drag the race's retention down."""
    top = [market_row(1, f"D{i}", fresh=114, rank=i + 1) for i in range(6)]
    tail = [market_row(1, f"T{i}", fresh=0, rank=7 + i, volume=100.0) for i in range(14)]
    out = sc.summarise(pd.DataFrame(top + tail))
    assert out["retention"].round(2).tolist() == [0.95]
    assert out["drivers"].tolist() == [6]


def price_rows(minutes_and_prices):
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime([f"2025-09-21T11:{m:02d}:00Z" for m, _ in minutes_and_prices]),
            "price": [p for _, p in minutes_and_prices],
            "size": [100.0] * len(minutes_and_prices),
        }
    )
    df["ts"] = df["ts"].dt.tz_convert("UTC")
    return df


def test_carry_forward_is_strictly_backward_and_expires():
    """A price covers later minutes within tolerance, never an earlier one."""
    grid = sc._epoch_minutes(price_rows([(m, 0.0) for m in range(0, 12)])["ts"])
    out = sc._carry_forward(price_rows([(5, 0.60)]), grid)
    assert pd.isna(out[4]), "a print must not price the minute before it"
    assert out[5] == 0.60
    assert out[10] == 0.60, "still inside the 5-minute tolerance"
    assert pd.isna(out[11]), "stale past tolerance"
