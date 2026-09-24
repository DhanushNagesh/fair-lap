"""Calibration of model and market on the same axes.

Equal-count bins, not equal-width: win probabilities pile up near 0, so
equal-width bins leave the top buckets with a handful of rows and a useless
curve.
"""

from __future__ import annotations

import pandas as pd


def calibration_table(
    df: pd.DataFrame, preds: list[str], n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Per bin: mean predicted, observed win rate, count, for each prediction column."""
    raise NotImplementedError


def plot_calibration(table: pd.DataFrame, out_path=None):
    """Model and market curves on one plot with the diagonal. Returns the figure."""
    raise NotImplementedError
