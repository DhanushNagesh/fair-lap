"""Phase 0: how many races actually have a usable in-race market?

For every 2023+ GP, report the share of race minutes with at least one price
point for each driver above the volume floor. The count of races clearing
MIN_MINUTE_COVERAGE is the size of the eval set, and that number goes in the
README before any modelling starts.

Output is a table (stdout) plus a CSV, so the number is reproducible rather
than remembered.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import pandas as pd


def race_window(session: pd.Series) -> tuple[int, int]:
    """(start_ts, end_ts) unix seconds for a race, from OpenF1 session bounds."""
    raise NotImplementedError


def coverage_for_market(yes_token_id: str, start_ts: int, end_ts: int) -> dict:
    """Minutes covered, minutes in window, first/last price ts, source used."""
    raise NotImplementedError


def scan(seasons: Iterable[int], min_volume_usd: float | None = None) -> pd.DataFrame:
    """One row per (race, driver) with coverage, plus a per-race rollup column."""
    raise NotImplementedError


def summarise(coverage: pd.DataFrame) -> pd.DataFrame:
    """Per-race summary: drivers above floor, median coverage, eval-set flag."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 market coverage scan")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--out", default="data/coverage.csv")
    args = parser.parse_args()
    coverage = scan(args.seasons, min_volume_usd=args.min_volume)
    coverage.to_csv(args.out, index=False)
    print(summarise(coverage).to_string(index=False))
