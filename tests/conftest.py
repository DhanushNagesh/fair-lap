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
