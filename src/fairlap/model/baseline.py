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

FROZEN_ANCHOR_LAP = 1


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
    """Baseline B. The de-vigged market price at the opening lap, held constant.

    Nothing is learned across races, so `fit` is a no-op: the anchor is lap-1
    data of the race being predicted, which is available at lap 1 by
    definition. Holding it flat is the whole point -- it is the market's
    opening opinion with no in-race updating, which is what the live model has
    to beat.

    A driver with no priced anchor lap gets NaN, not a guess. That driver drops
    out of the comparison in eval/ rather than being imputed into it.
    """

    def __init__(self, anchor_lap: int = FROZEN_ANCHOR_LAP) -> None:
        self.anchor_lap = anchor_lap

    def fit(self, df: pd.DataFrame) -> FrozenPreRaceBaseline:
        return self

    def predict_raw(self, df: pd.DataFrame) -> pd.Series:
        anchor = df.loc[df["lap_number"] == self.anchor_lap, :]
        anchor = (
            anchor[["session_key", "driver_number", "p_market"]]
            .dropna(subset=["p_market"])
            .drop_duplicates(subset=["session_key", "driver_number"], keep="first")
            .rename(columns={"p_market": "p_anchor"})
        )
        merged = df[["session_key", "driver_number"]].merge(
            anchor, on=["session_key", "driver_number"], how="left"
        )
        return pd.Series(merged["p_anchor"].to_numpy(), index=df.index, dtype="float64")

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return normalise_by_lap(df.assign(p_raw=self.predict_raw(df)))
