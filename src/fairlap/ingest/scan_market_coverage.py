"""Phase 0: how many races actually have a usable in-race market?

For every 2023+ GP, report the share of race minutes with at least one price
point for each driver above the volume floor. The count of races clearing
MIN_MINUTE_COVERAGE is the size of the eval set, and that number goes in the
README before any modelling starts.

Two coverage numbers are reported per market, deliberately:

  quote_coverage  minutes with a /prices-history point. That endpoint resamples
                  the CLOB, so it answers "was there a book" and is ~1.0 almost
                  everywhere, including markets nobody traded.
  trade_coverage  minutes with an actual fill. This answers "was there a price
                  somebody was willing to trade at", which is the question the
                  model-vs-market comparison actually rests on.
  fresh_coverage  minutes whose most recent fill is within
                  STALENESS_TOLERANCE_MIN. This is what a backward as-of join
                  with a staleness tolerance will actually find, so it is the
                  number that predicts how many (lap, driver) rows survive.

They measure different things and disagree by a lot. Which one gates the eval
set is a project decision, so the scan reports all three rather than picking.

Output is a table (stdout) plus a CSV, so the number is reproducible rather
than remembered.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np
import pandas as pd

from fairlap import db
from fairlap.config import (
    MARKET_ERA_START,
    MIN_MARKET_VOLUME_USD,
    MIN_MINUTE_COVERAGE,
    STALENESS_TOLERANCE_MIN,
    TOP_N_DRIVERS,
)
from fairlap.ingest import polymarket


def race_window(session: pd.Series) -> tuple[int, int]:
    """(start_ts, end_ts) unix seconds for a race, from OpenF1 session bounds."""
    start = pd.Timestamp(session["date_start"]).tz_convert("UTC")
    end = pd.Timestamp(session["date_end"]).tz_convert("UTC")
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise ValueError(f"session {session.get('session_key')} has no usable window")
    return int(start.timestamp()), int(end.timestamp())


def _epoch_minutes(ts: pd.Series) -> np.ndarray:
    """Minutes since epoch. Goes through seconds because the series may be
    second- or nanosecond-resolution depending on how it was built."""
    naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return (naive.astype("datetime64[s]").astype("int64") // 60).to_numpy()


def _fresh_minutes(
    minute_prices: pd.DataFrame, start_ts: int, end_ts: int, tolerance: int | None = None
) -> int:
    """Minutes in the window whose most recent fill is within `tolerance`.

    A strictly backward look-back, matching leakage rule 4: a minute is covered
    by a print that preceded it, never by the next one.
    """
    tolerance = STALENESS_TOLERANCE_MIN if tolerance is None else tolerance
    grid = np.arange(start_ts // 60, end_ts // 60 + 1)
    if minute_prices.empty:
        return 0
    fills = np.sort(_epoch_minutes(minute_prices["ts"]))
    idx = np.searchsorted(fills, grid, side="right") - 1
    age = np.where(idx >= 0, grid - fills[np.clip(idx, 0, None)], np.iinfo(np.int64).max)
    return int((age <= tolerance).sum())


def coverage_for_market(
    yes_token_id: str,
    condition_id: str,
    start_ts: int,
    end_ts: int,
    refresh: bool = False,
    client=None,
) -> dict:
    """Minutes covered, minutes in window, first/last price ts, source used."""
    # Count minute buckets, not elapsed seconds: a 119.5-minute session spans
    # 120 buckets, and dividing by 60 gave coverage fractions above 1.
    total_minutes = end_ts // 60 - start_ts // 60 + 1

    history = polymarket.fetch_prices_history(
        yes_token_id, start_ts, end_ts, refresh=refresh, client=client
    )
    quote_minutes = history["ts"].dt.floor("min").nunique() if not history.empty else 0

    trades = polymarket.fetch_trades(condition_id, start_ts, end_ts, refresh=refresh, client=client)
    minute_prices = polymarket.prices_from_trades(trades)
    trade_minutes = len(minute_prices)

    fresh_minutes = _fresh_minutes(minute_prices, start_ts, end_ts)

    if quote_minutes:
        source = "prices-history"
        first, last = history["ts"].min(), history["ts"].max()
    elif trade_minutes:
        source = "trades"
        first, last = minute_prices["ts"].min(), minute_prices["ts"].max()
    else:
        source = "none"
        first = last = pd.NaT

    return {
        "total_minutes": total_minutes,
        "quote_minutes": int(quote_minutes),
        "trade_minutes": int(trade_minutes),
        "fresh_minutes": int(fresh_minutes),
        "quote_coverage": quote_minutes / total_minutes,
        "trade_coverage": trade_minutes / total_minutes,
        "fresh_coverage": fresh_minutes / total_minutes,
        "trades_in_window": int(len(trades)),
        "traded_size": float(trades["size"].sum()) if not trades.empty else 0.0,
        "first_price_ts": first,
        "last_price_ts": last,
        "source": source,
    }


def _race_frame(seasons: Iterable[int]) -> pd.DataFrame:
    """Race sessions joined to their matched markets, straight from DuckDB."""
    con = db.connect(read_only=True)
    try:
        return con.execute(
            """
            SELECT s.session_key, s.year, s.circuit_short_name, s.country_name,
                   s.date_start, s.date_end, s.is_cancelled,
                   m.condition_id, m.yes_token_id, m.driver_name, m.volume_usd
            FROM raw_sessions s
            LEFT JOIN raw_market_meta m USING (session_key)
            WHERE s.year IN (SELECT UNNEST($1::INTEGER[]))
            ORDER BY s.date_start, m.volume_usd DESC
            """,
            [list(seasons)],
        ).fetchdf()
    finally:
        con.close()


def race_status(races: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Label every race session: is a market expected, and is one there?

    A race after MARKET_ERA_START that ran, was not cancelled, and has no
    matched market is a matcher failure, not a coverage fact. The two are
    indistinguishable in the output unless they are labelled here.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    out = (
        races.groupby("session_key", as_index=False)
        .agg(
            year=("year", "first"),
            circuit_short_name=("circuit_short_name", "first"),
            date_start=("date_start", "first"),
            is_cancelled=("is_cancelled", "first"),
            markets=("condition_id", "count"),
        )
        .sort_values("date_start")
    )
    start = pd.to_datetime(out["date_start"], utc=True)
    era = pd.Timestamp(MARKET_ERA_START, tz="UTC")

    ran = (start < now) & (~out["is_cancelled"].fillna(False))
    has_market = out["markets"] > 0

    out["status"] = "not_run"
    out.loc[ran & has_market, "status"] = "market_found"
    out.loc[ran & ~has_market & (start >= era), "status"] = "market_missing"
    out.loc[ran & ~has_market & (start < era), "status"] = "no_market_expected"
    return out


def scan(
    seasons: Iterable[int], min_volume_usd: float | None = None, refresh: bool = False
) -> pd.DataFrame:
    """One row per (race, driver) with coverage, plus a per-race rollup column."""
    floor = MIN_MARKET_VOLUME_USD if min_volume_usd is None else min_volume_usd
    races = _race_frame(seasons)
    status = race_status(races)
    scannable = set(status.loc[status["status"] == "market_found", "session_key"])

    markets = races[races["condition_id"].notna() & races["session_key"].isin(scannable)]
    rows = []
    with polymarket.session() as client:
        for n, (_, market) in enumerate(markets.iterrows(), start=1):
            start_ts, end_ts = race_window(market)
            stats = coverage_for_market(
                market["yes_token_id"],
                market["condition_id"],
                start_ts,
                end_ts,
                refresh=refresh,
                client=client,
            )
            if n % 50 == 0:
                print(f"  scanned {n}/{len(markets)} markets", flush=True)
            rows.append(
                {
                    "session_key": market["session_key"],
                    "year": market["year"],
                    "circuit_short_name": market["circuit_short_name"],
                    "race_date": pd.Timestamp(market["date_start"]).tz_convert("UTC").date(),
                    "driver_name": market["driver_name"],
                    "condition_id": market["condition_id"],
                    "yes_token_id": market["yes_token_id"],
                    "volume_usd": market["volume_usd"],
                    "above_volume_floor": bool((market["volume_usd"] or 0) >= floor),
                    **stats,
                }
            )

    coverage = pd.DataFrame(rows)
    if coverage.empty:
        return coverage

    # A market with no reported volume ranks last rather than dropping out of
    # the ranking entirely, which is what a NaN rank would do.
    coverage["volume_usd"] = coverage["volume_usd"].fillna(0.0)
    coverage["volume_rank"] = (
        coverage.groupby("session_key")["volume_usd"]
        .rank(ascending=False, method="first", na_option="bottom")
        .astype(int)
    )
    coverage["top_n"] = coverage["volume_rank"] <= TOP_N_DRIVERS
    return coverage.sort_values(["race_date", "volume_rank"])


def summarise(coverage: pd.DataFrame) -> pd.DataFrame:
    """Per-race summary: drivers above floor, median coverage, eval-set flag."""
    if coverage.empty:
        return pd.DataFrame()

    top = coverage[coverage["top_n"]]
    out = (
        top.groupby("session_key", as_index=False)
        .agg(
            race_date=("race_date", "first"),
            year=("year", "first"),
            circuit=("circuit_short_name", "first"),
            drivers=("driver_name", "size"),
            above_floor=("above_volume_floor", "sum"),
            median_quote_cov=("quote_coverage", "median"),
            median_trade_cov=("trade_coverage", "median"),
            median_fresh_cov=("fresh_coverage", "median"),
            min_fresh_cov=("fresh_coverage", "min"),
            trades=("trades_in_window", "sum"),
            volume_usd=("volume_usd", "sum"),
        )
        .sort_values("race_date")
    )
    out["in_eval_quote"] = out["median_quote_cov"] >= MIN_MINUTE_COVERAGE
    out["in_eval_trade"] = out["median_trade_cov"] >= MIN_MINUTE_COVERAGE
    out["in_eval_fresh"] = out["median_fresh_cov"] >= MIN_MINUTE_COVERAGE
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 market coverage scan")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--out", default="data/coverage.csv")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    races = _race_frame(args.seasons)
    status = race_status(races)
    print(status["status"].value_counts().to_string())

    missing = status[status["status"] == "market_missing"]
    if not missing.empty:
        print(f"\n{len(missing)} race(s) after {MARKET_ERA_START} with no matched market:")
        print(missing[["year", "circuit_short_name", "date_start"]].to_string(index=False))
        raise SystemExit("market_missing is non-empty: fix the event matcher before scanning")

    coverage = scan(args.seasons, min_volume_usd=args.min_volume, refresh=args.refresh)
    coverage.to_csv(args.out, index=False)
    summary = summarise(coverage)
    print()
    print(summary.to_string(index=False))
    threshold = f"{MIN_MINUTE_COVERAGE:.0%}"
    print(
        f"\nraces scanned: {len(summary)}"
        f"\neval set at quote coverage >= {threshold}: {int(summary['in_eval_quote'].sum())}"
        f"\neval set at trade coverage >= {threshold}: {int(summary['in_eval_trade'].sum())}"
        f"\neval set at fresh coverage (<= {STALENESS_TOLERANCE_MIN} min stale)"
        f" >= {threshold}: {int(summary['in_eval_fresh'].sum())}"
        f"\nCSV: {args.out} ({len(coverage)} market rows)"
    )
