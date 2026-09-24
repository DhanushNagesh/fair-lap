"""The two bars the real model has to clear.

Baseline A -- historical win rate by (position, lap_fraction bucket). Beating
this is the minimum; it knows nothing but where you are and how far in we are.

Baseline B -- pre-race market odds, frozen for the whole race. This is the one
that matters: beating it is the claim that live timing data adds information
the market's opening line did not already have.
"""

from __future__ import annotations

import pandas as pd


class PositionRateBaseline:
    """Baseline A. Empirical P(win | position, lap_fraction bucket)."""

    def __init__(self, n_lap_buckets: int = 10) -> None:
        raise NotImplementedError

    def fit(self, df: pd.DataFrame) -> PositionRateBaseline:
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class FrozenPreRaceBaseline:
    """Baseline B. The de-vigged market price at lap 1, held constant."""

    def fit(self, df: pd.DataFrame) -> FrozenPreRaceBaseline:
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
