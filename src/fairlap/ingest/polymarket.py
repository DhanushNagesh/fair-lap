"""Polymarket ingestion: market metadata, per-minute prices, raw trades.

Three endpoints, three jobs:

- Gamma /events?tag_id=435 -> the F1 events and their per-driver markets.
  Each driver is its own Yes/No market; clobTokenIds[0] is the Yes token.
- CLOB /prices-history?market=<yes_token_id>&startTs&endTs&fidelity=1 ->
  one point per minute. Known to return empty for some resolved markets
  (polymarket GitHub issue #216), hence the /trades fallback below.
- Data API /trades -> raw fills. Returns Yes and No trades mixed together:
  keep outcome == "Yes", or convert a No fill with 1 - p. Used for volume
  features, for the price fallback, and to cross-check prices-history.

Prices are stored raw (with vig). De-vigging happens in transform, because the
normalisation depends on which drivers made the volume cut.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import pandas as pd


def fetch_f1_events(seasons: Iterable[int]) -> pd.DataFrame:
    """F1 race-winner events from Gamma, one row per market (driver)."""
    raise NotImplementedError


def match_events_to_sessions(events: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Map each Polymarket event to an OpenF1 session_key.

    Matched on circuit/country plus event date, not on title text -- titles are
    inconsistent across seasons ("Azerbaijan GP" vs "Baku GP"). Unmatched
    events are returned with a null session_key rather than dropped, so the
    coverage scan can report them.
    """
    raise NotImplementedError


def fetch_prices_history(yes_token_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Per-minute Yes price for one driver market (fidelity=1)."""
    raise NotImplementedError


def fetch_trades(yes_token_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Raw fills for one market, Yes and No mixed, as returned by the API."""
    raise NotImplementedError


def prices_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a per-minute Yes price series from fills.

    Fallback for markets where prices-history is empty. No trades in a minute
    means no row -- never forward-fill here; a stale price must be visibly
    absent so eval can exclude the minute.
    """
    raise NotImplementedError


def ingest(seasons: Iterable[int], min_volume_usd: float | None = None) -> dict[str, int]:
    """Load market metadata, prices and trades for markets above the volume floor."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Polymarket F1 odds into DuckDB")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(ingest(args.seasons, min_volume_usd=args.min_volume))
