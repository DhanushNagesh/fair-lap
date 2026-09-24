"""The headline comparison: model vs. market, paired.

Only rows where both sides have a valid prediction enter the comparison -- same
(session_key, lap_number, driver_number), same de-vigged basis. Minutes with no
trade are dropped rather than forward-filled, because beating a stale quote is
not beating the market.

Done when this module can fill in: "the model beats/loses to the market in
___, because ___."
"""

from __future__ import annotations

import argparse

import pandas as pd


def load_paired(seasons: list[int] | None = None) -> pd.DataFrame:
    """Read predictions and features from DuckDB and return the paired frame."""
    raise NotImplementedError


def paired_frame(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Inner join model and market predictions on (session, lap, driver)."""
    raise NotImplementedError


def paired_comparison(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Brier difference (model - market) per group with race-clustered bootstrap CIs."""
    raise NotImplementedError


def betting_backtest(
    df: pd.DataFrame, edge_threshold: float = 0.03, spread: float = 0.01, fee: float = 0.0
) -> pd.DataFrame:
    """Flat-stake backtest on rows where the model disagrees past the threshold.

    Reported whether or not it makes money. A positive PnL here is weak
    evidence -- the sample is a few dozen races and the spread assumption is a
    guess -- so it is a sanity check on the Brier result, not the result.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the model against the market")
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()
    result = paired_comparison(load_paired(args.seasons), by=["phase", "condition"])
    print(result.to_string(index=False))
    if args.backtest:
        print(betting_backtest(load_paired(args.seasons)).to_string(index=False))
