"""Build the lap-level feature table.

One row per (session_key, lap_number, driver_number), using only information
known at the end of that lap. Anything computed from a later lap -- final
classification, total stops, stint length in hindsight -- is leakage and is
tested against in tests/test_leakage.py.

Feature groups:
  race state     position, gap_to_leader_s, gap_to_ahead_s, lap_fraction
  tyres          compound, tyre_age_laps, stops_made,
                 expected_remaining_stops (pit loss not yet paid -- the
                 feature the whole model leans on)
  interruptions  sc_active, vsc_active, red_flag_active, laps_since_sc
  prior          grid_position, quali_gap_s, season
  market         p_market (as-of, de-vigged), market_overround
  target         won (0/1)
"""

from __future__ import annotations

import argparse

import pandas as pd

FEATURE_COLUMNS = (
    "position",
    "gap_to_leader_s",
    "gap_to_ahead_s",
    "lap_fraction",
    "compound",
    "tyre_age_laps",
    "stops_made",
    "expected_remaining_stops",
    "sc_active",
    "vsc_active",
    "red_flag_active",
    "laps_since_sc",
    "grid_position",
    "quali_gap_s",
    "season",
)
TARGET = "won"


def lap_frame(session_key: int) -> pd.DataFrame:
    """Skeleton of (lap, driver) rows with lap_start/lap_end timestamps."""
    raise NotImplementedError


def add_race_state(df: pd.DataFrame) -> pd.DataFrame:
    """position, gaps and lap_fraction as of the end of each lap."""
    raise NotImplementedError


def add_tyre_state(df: pd.DataFrame) -> pd.DataFrame:
    """compound, tyre_age_laps, stops_made and expected_remaining_stops.

    expected_remaining_stops comes from a per-circuit stop-count prior and the
    laps remaining, NOT from how many stops the driver actually went on to
    make. Getting this wrong is the single easiest way to leak the result.
    """
    raise NotImplementedError


def add_interruptions(df: pd.DataFrame) -> pd.DataFrame:
    """SC / VSC / red-flag flags and laps_since_sc from race_control messages."""
    raise NotImplementedError


def add_prior(df: pd.DataFrame) -> pd.DataFrame:
    """grid_position and quali gap, known before the race starts."""
    raise NotImplementedError


def add_market(df: pd.DataFrame) -> pd.DataFrame:
    """De-vigged market price attached by backward as-of join on lap_end."""
    raise NotImplementedError


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """won = 1 for the race winner on every one of that driver's laps."""
    raise NotImplementedError


def build(session_keys: list[int] | None = None) -> pd.DataFrame:
    """Full feature table; writes to the `features` table in DuckDB."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the lap-level feature table")
    parser.add_argument("--session-key", type=int, nargs="*", default=None)
    args = parser.parse_args()
    print(len(build(args.session_key)), "rows written")
