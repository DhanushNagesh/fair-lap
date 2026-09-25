"""De-vigging: the market must be normalised before anything is compared to it."""

from __future__ import annotations

import pandas as pd
import pytest

from fairlap.config import MIN_DEVIG_DRIVERS, OVERROUND_BAND
from fairlap.transform.asof import devig


def test_devig_sums_to_one_per_minute(minute_prices):
    out = devig(minute_prices)
    per_minute = out.groupby("ts")["p_market"].sum()
    assert per_minute.round(9).eq(1.0).all()


def test_devig_records_overround(minute_prices):
    """0.60 + 0.45 = 1.05, so overround must come back as 1.05."""
    out = devig(minute_prices)
    first = out[out["ts"] == out["ts"].min()]
    assert first["overround"].unique().tolist() == [pytest.approx(1.05)]


def test_devig_preserves_relative_odds(minute_prices):
    """Normalising may not reorder the field or change price ratios."""
    out = devig(minute_prices)
    first = out[out["ts"] == out["ts"].min()].set_index("driver_number")
    assert first.loc[1, "p_market"] / first.loc[44, "p_market"] == pytest.approx(0.60 / 0.45)


def test_a_minute_with_too_few_drivers_is_not_deviggable(minute_prices):
    """One lone price has nothing to normalise against, so it scores nothing."""
    lonely = minute_prices[minute_prices["driver_number"] == 1]
    out = devig(lonely)
    assert MIN_DEVIG_DRIVERS > 1
    assert out["p_market"].isna().all()
    # The diagnostic survives, so the reason is visible in the output.
    assert out["overround"].notna().all()


def test_a_minute_far_outside_the_overround_band_is_rejected(minute_prices):
    """An overround of 2.0 means drivers are missing, not that there is an edge."""
    broken = minute_prices.copy()
    broken.loc[broken["ts"] == broken["ts"].min(), "price"] = 1.0
    out = devig(broken)
    first = out[out["ts"] == out["ts"].min()]
    assert first["overround"].iloc[0] > OVERROUND_BAND[1]
    assert first["p_market"].isna().all()
    # The other minute is untouched: the rejection is per minute, not global.
    assert out[out["ts"] == out["ts"].max()]["p_market"].notna().all()


def test_devig_is_applied_per_minute_not_across_the_race(minute_prices):
    """Two minutes normalise independently; pooling them would mix timestamps."""
    out = devig(minute_prices)
    assert out.groupby("ts")["overround"].nunique().eq(1).all()
    assert out["overround"].nunique() >= 1
    assert out.groupby("ts")["p_market"].sum().round(9).eq(1.0).all()


def test_devig_on_an_empty_frame_returns_the_columns(minute_prices):
    out = devig(minute_prices.iloc[0:0])
    assert {"p_market", "overround"} <= set(out.columns)
    assert out.empty


def test_devig_ignores_a_null_price_rather_than_treating_it_as_zero():
    """A driver with no print must not drag the denominator down."""
    frame = pd.DataFrame(
        {
            "session_key": [1, 1, 1],
            "ts": pd.to_datetime(["2025-01-01T12:00:00Z"] * 3),
            "driver_number": [1, 2, 3],
            "price": [0.60, 0.45, None],
        }
    )
    out = devig(frame)
    assert out["overround"].dropna().unique().tolist() == [pytest.approx(1.05)]
    assert pd.isna(out.loc[out["driver_number"] == 3, "p_market"].iloc[0])
    assert out["p_market"].sum() == pytest.approx(1.0)
