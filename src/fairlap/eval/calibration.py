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


# Display names for the prediction columns, so the legend reads as the results
# table does rather than as column names. Insertion order is the plotting and
# legend order: the market first because everything else is measured against it.
LABELS = {
    "p_market": "market (de-vigged)",
    "p_gbm": "GBM",
    "p_logistic": "logistic",
    "p_position_rate": "baseline A: position rate",
    "p_frozen_prerace": "baseline B: frozen closing line",
}
# Where the second panel stops. Eight of ten quantile bins sit below this, so a
# single linear panel collapses most of both curves onto the origin.
ZOOM_MAX = 0.15


def _series_order(table: pd.DataFrame) -> list[str]:
    """Known columns in LABELS order, then anything else alphabetically."""
    present = list(dict.fromkeys(table["pred"]))
    known = [c for c in LABELS if c in present]
    return known + sorted(c for c in present if c not in LABELS)


def plot_calibration(table: pd.DataFrame, out_path=None, zoom: float = ZOOM_MAX):
    """Every forecaster's curve against the diagonal. Returns the figure.

    Two panels, because one cannot show this data. Win probabilities pile up
    near zero: on a full [0, 1] axis eight of the ten quantile bins land in the
    leftmost tenth and the interesting part -- whether a forecaster's
    write-offs actually lose -- is invisible. The right panel is the same curves
    over [0, `zoom`].

    matplotlib is an optional dependency (`uv sync --all-extras`); `make eval`
    prints the table and does not need the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plot_calibration needs matplotlib: uv sync --all-extras") from exc

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, hi in zip(axes, (1.0, zoom), strict=True):
        ax.plot([0, hi], [0, hi], color="0.55", linewidth=1, linestyle="--", zorder=1)
        for col in _series_order(table):
            part = table[table["pred"] == col]
            # The market is the benchmark, not one of the contestants, so it is
            # drawn in black and heavier than the models being measured.
            is_market = col == "p_market"
            ax.plot(
                part["mean_pred"],
                part["observed"],
                marker="o",
                markersize=4,
                linewidth=2.4 if is_market else 1.4,
                color="black" if is_market else None,
                alpha=1.0 if is_market else 0.85,
                label=LABELS.get(str(col), str(col)),
                zorder=3 if is_market else 2,
            )
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed win rate")
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[0].set_title("full range")
    axes[1].set_title(f"low-probability region (below {zoom:g})")
    # Legend on the left panel only; the right one is the same five series.
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig
