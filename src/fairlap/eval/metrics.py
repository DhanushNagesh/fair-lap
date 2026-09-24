"""Brier score, log loss, and the breakdowns that make them informative.

A single aggregate Brier over every lap is dominated by the late-race rows
where the winner is obvious and everyone scores well. The phase and condition
breakdowns are the actual result.
"""

from __future__ import annotations

import pandas as pd

PHASES = {"opening": (1, 10), "middle": (11, -11), "closing": (-10, None)}


def brier(y_true: pd.Series, p: pd.Series) -> float:
    raise NotImplementedError


def log_loss(y_true: pd.Series, p: pd.Series, eps: float = 1e-15) -> float:
    raise NotImplementedError


def label_phase(df: pd.DataFrame) -> pd.Series:
    """opening / middle / closing by lap number relative to total_laps."""
    raise NotImplementedError


def score_by(df: pd.DataFrame, by: list[str], preds: list[str]) -> pd.DataFrame:
    """Brier and log loss per group, one column block per prediction column."""
    raise NotImplementedError


def bootstrap_ci(
    df: pd.DataFrame,
    stat_fn,
    n_boot: int = 2_000,
    cluster: str = "session_key",
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile CI, resampling whole races.

    Clustering on race is not optional: laps within a race are near-perfectly
    correlated, so a row-level bootstrap would report intervals several times
    too narrow and turn noise into a headline.
    """
    raise NotImplementedError
