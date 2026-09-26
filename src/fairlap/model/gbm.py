"""Logistic regression and LightGBM on the lap-level feature table.

Both are trained per-row as a binary classifier and then renormalised so each
(session_key, lap_number) sums to 1 across the field. Renormalising a
per-driver classifier is the defensible choice over a softmax-over-field
ranker: the field size varies with retirements, and a plain binary model keeps
per-driver calibration interpretable. The cost is that raw outputs are not
coherent as a distribution until normalise_by_lap runs.
"""

from __future__ import annotations

import pandas as pd

CATEGORICAL = ("compound",)


def normalise_by_lap(df: pd.DataFrame, col: str = "p_raw") -> pd.Series:
    """Scale probabilities so each lap sums to 1 over the drivers still running.

    "Still running" needs no retirement column: the feature table only has a
    row for a lap a driver actually completed, so the rows present at
    (session_key, lap_number) are exactly the surviving field.

    Nulls stay null and are left out of the denominator. That is deliberate for
    the frozen market baseline, where an unpriced driver has no opinion to
    renormalise -- filling them with a number would invent a quote. A lap whose
    raw values sum to zero also stays null rather than dividing by zero.
    """
    p = pd.to_numeric(df[col], errors="coerce")
    total = p.groupby([df["session_key"], df["lap_number"]]).transform("sum")
    return p.div(total.where(total > 0))


class LogisticModel:
    """Linear baseline; the interpretable reference for the GBM's coefficients."""

    def fit(self, df: pd.DataFrame) -> LogisticModel:
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class GBMModel:
    """LightGBM binary classifier, grouped by race for any CV."""

    def __init__(self, **params) -> None:
        raise NotImplementedError

    def fit(self, df: pd.DataFrame) -> GBMModel:
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def save(self, path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> GBMModel:
        raise NotImplementedError
