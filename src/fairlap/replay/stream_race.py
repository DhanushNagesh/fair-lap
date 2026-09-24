"""Stream a past race one lap at a time.

The replay is the honesty mechanism: it hands the model a state object built
only from laps <= t, so a feature that needs the future cannot be computed at
all rather than quietly computing a leaked value. If `build_features` and the
replay ever disagree on a row, the replay is right.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LapState:
    """Everything known at the end of one lap."""

    session_key: int
    lap_number: int
    total_laps: int
    lap_end: pd.Timestamp
    drivers: pd.DataFrame  # one row per driver, FEATURE_COLUMNS subset


def stream(session_key: int, speed: float | None = None) -> Iterator[LapState]:
    """Yield LapState for each lap in order.

    `speed` sleeps between laps to mimic real timing for the dashboard demo;
    None runs as fast as the data loads.
    """
    raise NotImplementedError


def run(session_key: int, model_name: str = "gbm", speed: float | None = None) -> pd.DataFrame:
    """Replay a race, predict at each lap, write to the `predictions` table."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a race lap by lap")
    parser.add_argument("--session-key", type=int, required=True)
    parser.add_argument("--model", default="gbm")
    parser.add_argument("--speed", type=float, default=None)
    args = parser.parse_args()
    run(args.session_key, model_name=args.model, speed=args.speed)
