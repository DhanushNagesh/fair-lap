"""Brier score, log loss, and the breakdowns that make them informative.

A single aggregate Brier over every lap is dominated by the late-race rows
where the winner is obvious and everyone scores well. The phase and condition
breakdowns are the actual result.

Every function here drops rows where either the outcome or the prediction is
null rather than imputing. A model with no opinion on a row does not get a
0.5 handed to it, because that would be a number nobody produced.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

# Phase boundaries in laps. A lap is "opening" if it is in the first ten and
# "closing" if it is in the last ten; a race shorter than twenty laps (only a
# red-flagged one) resolves the overlap in favour of opening, since lap 3 of a
# 15-lap race is an opening lap that happens also to be near the end.
OPENING_LAPS = 10
CLOSING_LAPS = 10
PHASE_ORDER = ("opening", "middle", "closing")
CONDITION_ORDER = ("green", "interrupted")


def _aligned(y_true: pd.Series, p: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Both series as float arrays, restricted to rows where both are present."""
    y = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors="coerce")
    q = pd.to_numeric(pd.Series(p).reset_index(drop=True), errors="coerce")
    keep = y.notna() & q.notna()
    return y[keep].to_numpy(dtype="float64"), q[keep].to_numpy(dtype="float64")


def brier(y_true: pd.Series, p: pd.Series) -> float:
    """Mean squared error of the probability. Lower is better."""
    y, q = _aligned(y_true, p)
    if y.size == 0:
        return float("nan")
    return float(np.mean((q - y) ** 2))


def log_loss(y_true: pd.Series, p: pd.Series, eps: float = 1e-15) -> float:
    """Mean negative log likelihood, with probabilities clipped off 0 and 1.

    The clip is why log loss is reported next to Brier rather than instead of
    it: one confident miss at p=0 would otherwise be infinite, so the metric's
    worst case is set by `eps` rather than by the data.
    """
    y, q = _aligned(y_true, p)
    if y.size == 0:
        return float("nan")
    q = np.clip(q, eps, 1 - eps)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def label_phase(df: pd.DataFrame) -> pd.Series:
    """opening / middle / closing by lap number relative to total_laps."""
    lap = pd.to_numeric(df["lap_number"], errors="coerce")
    total = pd.to_numeric(df["total_laps"], errors="coerce")
    phase = pd.Series("middle", index=df.index, dtype="object")
    phase[lap > total - CLOSING_LAPS] = "closing"
    phase[lap <= OPENING_LAPS] = "opening"
    phase[lap.isna() | total.isna()] = None
    return pd.Categorical(phase, categories=PHASE_ORDER, ordered=True)


def label_condition(df: pd.DataFrame) -> pd.Series:
    """green vs interrupted, where interrupted is SC, VSC or red flag.

    The three are pooled because separating them splits a few hundred rows
    three ways. They share the mechanism that matters: the field compresses,
    the pit-stop calculus changes, and the market starts trading again.
    """
    flags = ["sc_active", "vsc_active", "red_flag_active"]
    present = [c for c in flags if c in df.columns]
    if not present:
        raise KeyError(f"frame carries none of {flags}")
    interrupted = df[present].fillna(False).astype(bool).any(axis=1)
    return pd.Categorical(
        np.where(interrupted, "interrupted", "green"), categories=CONDITION_ORDER, ordered=True
    )


def with_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach phase and condition, the two breakdown columns eval groups by."""
    return df.assign(phase=label_phase(df), condition=label_condition(df))


def score_by(df: pd.DataFrame, by: Sequence[str], preds: Sequence[str]) -> pd.DataFrame:
    """Brier and log loss per group, one column block per prediction column.

    `races` is reported beside `n` on purpose: the effective sample size is
    races, not rows, so a group with 4,000 rows from three races is a group
    with three observations in it.
    """
    by = list(by)
    groups: list[tuple] = [((), df)] if not by else list(df.groupby(by, observed=True))
    rows = []
    for key, part in groups:
        row: dict = {}
        if by:
            values = key if isinstance(key, tuple) else (key,)
            row.update(dict(zip(by, values, strict=True)))
        row["n"] = len(part)
        row["races"] = part["session_key"].nunique()
        row["base_rate"] = float(pd.to_numeric(part["won"], errors="coerce").mean())
        for col in preds:
            row[f"{col}_n"] = int(part[col].notna().sum())
            row[f"{col}_brier"] = brier(part["won"], part[col])
            row[f"{col}_logloss"] = log_loss(part["won"], part[col])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 2_000,
    cluster: str = "session_key",
    seed: int | None = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI, resampling whole races.

    Clustering on race is not optional: laps within a race are near-perfectly
    correlated, so a row-level bootstrap would report intervals several times
    too narrow and turn noise into a headline.

    One resample draws `n_races` race keys with replacement and concatenates
    every row of each drawn race -- a race drawn twice contributes all its laps
    twice. That is the point: it reproduces the fact that one unusual race
    moves the statistic by a whole race's worth, not by one row's worth.
    """
    if df.empty:
        return (float("nan"), float("nan"))
    keys = df[cluster].to_numpy()
    positions = {k: np.flatnonzero(keys == k) for k in pd.unique(keys)}
    race_keys = np.array(list(positions))
    rng = np.random.default_rng(seed)

    stats = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        drawn = rng.choice(race_keys, size=race_keys.size, replace=True)
        idx = np.concatenate([positions[k] for k in drawn])
        stats[i] = stat_fn(df.take(idx))

    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def brier_diff(model_col: str, market_col: str = "p_market") -> Callable[[pd.DataFrame], float]:
    """A stat_fn for `bootstrap_ci`: model Brier minus market Brier.

    Negative means the model is better. Paired by construction -- both scores
    come from the same rows of the same resampled races, so the difference is
    not two independent noisy numbers subtracted.
    """

    def stat(part: pd.DataFrame) -> float:
        return brier(part["won"], part[model_col]) - brier(part["won"], part[market_col])

    return stat
