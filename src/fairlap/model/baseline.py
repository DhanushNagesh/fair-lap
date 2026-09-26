"""The two bars the real model has to clear.

Baseline A -- historical win rate by (position, lap_fraction bucket). Beating
this is the minimum; it knows nothing but where you are and how far in we are.

Baseline B -- pre-race market odds, frozen for the whole race. This is the one
that matters: beating it is the claim that live timing data adds information
the market's opening line did not already have.

Both return per-lap normalised probabilities from `predict_proba`. The raw
per-driver number is available from `predict_raw` for inspection, but it is not
a distribution and is never what gets scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairlap.model.gbm import normalise_by_lap

ANCHOR_COLUMN = "p_market_prerace"


class PositionRateBaseline:
    """Baseline A. Empirical P(win | position, lap_fraction bucket).

    A lookup table, not a model: every (position, phase-of-race) cell gets the
    fraction of training rows in that cell that went on to win. Cells with few
    rows are shrunk toward that lap bucket's overall win rate, so a position
    seen three times in training does not report 1.0 or 0.0 with confidence.
    """

    def __init__(
        self, n_lap_buckets: int = 10, max_position: int = 20, prior_strength: float = 20.0
    ) -> None:
        self.n_lap_buckets = n_lap_buckets
        self.max_position = max_position
        self.prior_strength = prior_strength
        self.rates_: pd.Series | None = None
        self.bucket_rates_: pd.Series | None = None
        self.global_rate_: float = 0.0

    def _cells(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        bucket = (
            np.ceil(df["lap_fraction"].clip(lower=1e-9, upper=1.0) * self.n_lap_buckets)
            .clip(1, self.n_lap_buckets)
            .astype("Int64")
        )
        position = df["position"].clip(upper=self.max_position).round().astype("Int64")
        return bucket.rename("lap_bucket"), position.rename("pos")

    def fit(self, df: pd.DataFrame) -> PositionRateBaseline:
        d = df.dropna(subset=["position", "lap_fraction", "won"])
        bucket, position = self._cells(d)
        won = d["won"].astype(float)

        self.global_rate_ = float(won.mean())
        self.bucket_rates_ = won.groupby(bucket).mean()

        cells = won.groupby([bucket, position]).agg(["sum", "count"])
        prior_rate = cells.index.get_level_values("lap_bucket").map(self.bucket_rates_)
        # Shrink toward the bucket's own rate, not the global one: a rear-grid
        # position late in a race is genuinely near zero, and the global rate
        # would drag it up.
        self.rates_ = (cells["sum"] + self.prior_strength * prior_rate) / (
            cells["count"] + self.prior_strength
        )
        return self

    def predict_raw(self, df: pd.DataFrame) -> pd.Series:
        if self.rates_ is None:
            raise RuntimeError("fit() before predict")
        bucket, position = self._cells(df)
        keys = pd.MultiIndex.from_arrays([bucket, position])
        p = pd.Series(self.rates_.reindex(keys).to_numpy(), index=df.index, dtype="float64")
        # An unseen cell falls back to the lap bucket, then to the global rate.
        # A row with no position at all gets the global rate rather than a null
        # that would silently drop the driver out of the lap's denominator.
        fallback = pd.Series(
            bucket.map(self.bucket_rates_).to_numpy(), index=df.index, dtype="float64"
        )
        return p.fillna(fallback).fillna(self.global_rate_)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return normalise_by_lap(df.assign(p_raw=self.predict_raw(df)))


class FrozenPreRaceBaseline:
    """Baseline B. The de-vigged pre-race closing line, held constant.

    Reads `p_market_prerace` straight from the feature table -- the driver's
    last fill in the hour before lights out, de-vigged across the grid. It is
    already one constant per (race, driver), so "frozen" is a matter of not
    updating it rather than of picking a lap to freeze at.

    Renormalising per lap is what makes it a real opponent rather than a fixed
    vector: as drivers retire they leave the lap's denominator and the
    survivors' probabilities rise. That is the market's opening opinion updated
    with nothing except who is still running.

    Nothing is learned across races, so `fit` is a no-op.

    A driver with no anchor gets NaN, not a guess -- they drop out of the
    comparison in eval/ rather than being imputed into it.
    """

    def __init__(self, column: str = ANCHOR_COLUMN) -> None:
        self.column = column

    def fit(self, df: pd.DataFrame) -> FrozenPreRaceBaseline:
        return self

    def predict_raw(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            raise KeyError(f"{self.column} is not in the frame; rebuild the feature table")
        return pd.to_numeric(df[self.column], errors="coerce")

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return normalise_by_lap(df.assign(p_raw=self.predict_raw(df)))
