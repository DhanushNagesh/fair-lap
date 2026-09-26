"""Calibration of model and market on the same axes.

Equal-count bins, not equal-width: win probabilities pile up near 0, so
equal-width bins leave the top buckets with a handful of rows and a useless
curve. With ~15k rows and a 1-in-20 base rate, an equal-width top decile can
hold single digits of rows; its observed rate is then 0 or 1 and the curve
swings wildly for reasons that have nothing to do with the forecaster.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

QUANTILE = "quantile"
UNIFORM = "uniform"


def _bin(p: pd.Series, n_bins: int, strategy: str) -> pd.Series:
    if strategy == QUANTILE:
        # duplicates="drop" because a mass of identical probabilities (a whole
        # lap of back-markers at the same near-zero value) makes two quantile
        # edges equal, which qcut otherwise refuses outright.
        return pd.qcut(p, n_bins, labels=False, duplicates="drop")
    if strategy == UNIFORM:
        return pd.cut(p, n_bins, labels=False, include_lowest=True)
    raise ValueError(f"strategy must be {QUANTILE!r} or {UNIFORM!r}, got {strategy!r}")


def calibration_table(
    df: pd.DataFrame,
    preds: Sequence[str],
    n_bins: int = 10,
    strategy: str = QUANTILE,
    target: str = "won",
) -> pd.DataFrame:
    """Per bin: mean predicted, observed win rate, count, for each prediction column.

    Long format -- one row per (pred, bin) -- because each column is binned on
    its own quantiles. Sharing one set of edges across model and market would
    put different numbers of rows in the same bin and make the two curves
    incomparable at the level that matters, which is per-bin count.
    """
    rows = []
    for col in preds:
        part = df[[col, target]].dropna()
        if part.empty:
            continue
        p = pd.to_numeric(part[col], errors="coerce")
        y = pd.to_numeric(part[target], errors="coerce")
        bins = _bin(p, n_bins, strategy)
        for b, idx in p.groupby(bins).groups.items():
            rows.append(
                {
                    "pred": col,
                    "bin": int(b),
                    "n": len(idx),
                    "p_low": float(p.loc[idx].min()),
                    "p_high": float(p.loc[idx].max()),
                    "mean_pred": float(p.loc[idx].mean()),
                    "observed": float(y.loc[idx].mean()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["gap"] = out["observed"] - out["mean_pred"]
    return out.sort_values(["pred", "bin"], ignore_index=True)


def expected_calibration_error(table: pd.DataFrame) -> pd.Series:
    """Row-weighted mean |observed - predicted| per prediction column.

    A summary of the curve, not a replacement for it: ECE hides direction, and
    a forecaster that is 5 points high at the bottom and 5 points low at the
    top scores the same as one that is uniformly 5 points off.
    """
    if table.empty:
        return pd.Series(dtype="float64")
    weighted = table.assign(w=table["n"] * table["gap"].abs())
    return weighted.groupby("pred")["w"].sum() / table.groupby("pred")["n"].sum()


def plot_calibration(table: pd.DataFrame, out_path=None):
    """Model and market curves on one plot with the diagonal. Returns the figure.

    matplotlib is an optional dependency (`uv sync --all-extras`); `make eval`
    prints the table and never needs the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plot_calibration needs matplotlib: uv sync --all-extras") from exc

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=1, linestyle="--", label="perfect")
    for col, part in table.groupby("pred"):
        ax.plot(part["mean_pred"], part["observed"], marker="o", label=col)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig
