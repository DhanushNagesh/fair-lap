"""Shared fixtures. Tests never hit the network -- fixtures are hand-built frames."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def two_lap_race() -> pd.DataFrame:
    """Minimal (lap, driver) frame: 2 drivers, 3 laps, driver 1 wins."""
    return pd.DataFrame(
        {
            "session_key": [9999] * 6,
            "lap_number": [1, 1, 2, 2, 3, 3],
            "driver_number": [1, 44, 1, 44, 1, 44],
            "total_laps": [3] * 6,
            "position": [1, 2, 2, 1, 1, 2],
            "lap_end": pd.to_datetime(
                [
                    "2025-09-21T11:01:00Z",
                    "2025-09-21T11:01:01Z",
                    "2025-09-21T11:02:00Z",
                    "2025-09-21T11:02:01Z",
                    "2025-09-21T11:03:00Z",
                    "2025-09-21T11:03:01Z",
                ]
            ),
            "won": [1, 0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def minute_prices() -> pd.DataFrame:
    """Two driver price series, one minute apart, deliberately summing above 1."""
    return pd.DataFrame(
        {
            "session_key": [9999] * 4,
            "driver_number": [1, 44, 1, 44],
            "ts": pd.to_datetime(
                [
                    "2025-09-21T11:00:00Z",
                    "2025-09-21T11:00:00Z",
                    "2025-09-21T11:02:30Z",
                    "2025-09-21T11:02:30Z",
                ]
            ),
            "price": [0.60, 0.45, 0.80, 0.25],
        }
    )


def _ts(*values) -> pd.Series:
    return pd.to_datetime(list(values), utc=True).astype("datetime64[ns, UTC]")


@pytest.fixture
def synthetic_race():
    """A hand-built 2-driver, 5-lap race, as a RaceInputs.

    Deliberately awkward: driver 1 pits into a second stint on lap 3, a safety
    car is deployed on lap 2 and withdrawn on lap 4, and the market trades at
    three separate minutes. Every branch the leakage tests care about is
    reachable without touching DuckDB.
    """
    from fairlap.transform.race_inputs import RaceInputs

    base = pd.Timestamp("2025-01-01T12:00:00Z")
    rows = []
    for lap in range(1, 6):
        for driver in (1, 44):
            rows.append(
                {
                    "session_key": 9999,
                    "driver_number": driver,
                    "lap_number": lap,
                    "date_start": base
                    + pd.Timedelta(minutes=lap - 1)
                    + pd.Timedelta(seconds=driver),
                    "lap_duration": 60.0,
                }
            )
    laps = pd.DataFrame(rows)
    laps["date_start"] = laps["date_start"].astype("datetime64[ns, UTC]")

    position = pd.DataFrame(
        {
            "session_key": 9999,
            "driver_number": [1, 44, 1, 44],
            "date": _ts(
                "2025-01-01T11:59:00Z",
                "2025-01-01T11:59:00Z",
                "2025-01-01T12:03:30Z",
                "2025-01-01T12:03:30Z",
            ),
            "position": [1, 2, 2, 1],
        }
    )
    intervals = pd.DataFrame(
        {
            "session_key": 9999,
            "driver_number": [1, 44],
            "date": _ts("2025-01-01T12:01:30Z", "2025-01-01T12:01:30Z"),
            "gap_to_leader": ["0.0", "1.5"],
            "interval": ["0.0", "1.5"],
        }
    )
    stints = pd.DataFrame(
        {
            "session_key": 9999,
            "driver_number": [1, 1, 44],
            "stint_number": [1, 2, 1],
            "lap_start": [1, 3, 1],
            "compound": ["SOFT", "HARD", "MEDIUM"],
            "tyre_age_at_start": [0, 0, 2],
        }
    )
    pit = pd.DataFrame({"session_key": [9999], "driver_number": [1], "lap_number": [3]})
    race_control = pd.DataFrame(
        {
            "session_key": 9999,
            "date": _ts("2025-01-01T12:01:30Z", "2025-01-01T12:03:30Z"),
            "category": ["SafetyCar", "SafetyCar"],
            "message": ["SAFETY CAR DEPLOYED", "SAFETY CAR IN THIS LAP"],
            "flag": [None, None],
            "scope": [None, None],
            "lap_number": [2, 4],
        }
    )
    market = pd.DataFrame(
        {
            "session_key": 9999,
            "driver_number": [1, 44, 1, 44],
            "ts": _ts(
                "2025-01-01T12:00:00Z",
                "2025-01-01T12:00:00Z",
                "2025-01-01T12:03:00Z",
                "2025-01-01T12:03:00Z",
            ),
            "price": [0.60, 0.45, 0.80, 0.25],
        }
    )
    return RaceInputs(
        session_key=9999,
        year=2025,
        circuit_key=7,
        circuit_short_name="Testing",
        total_laps=5,
        laps=laps,
        position=position,
        intervals=intervals,
        stints=stints,
        pit=pit,
        race_control=race_control,
        market=market,
    )
