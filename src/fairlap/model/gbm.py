"""Logistic regression and LightGBM on the lap-level feature table.

Both are trained per-row as a binary classifier and then renormalised so each
(session_key, lap_number) sums to 1 across the field. Renormalising a
per-driver classifier is the defensible choice over a softmax-over-field
ranker: the field size varies with retirements, and a plain binary model keeps
per-driver calibration interpretable. The cost is that raw outputs are not
coherent as a distribution until normalise_by_lap runs.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = (
    "position",
    "grid_position",
    "gap_to_leader_s",
    "gap_to_ahead_s",
    "lap_time_s",
    "pace_roll_s",
    "lap_fraction",
    "laps_remaining",
    "tyre_age_laps",
    "stops_made",
    "expected_remaining_stops",
    "laps_since_sc",
)
# stint_number is in the feature table but not here: it is stops_made + 1, and
# the two correlate at 0.98. Carrying both splits one signal across two
# coefficients with opposite signs and makes the linear model useless as the
# sanity check on the GBM, which is the only reason it is being fit.
#
# gap_to_ahead_s is kept, but read it knowing OpenF1 reports it as exactly 0.00
# for P1. It therefore doubles as a leader indicator and its coefficient is not
# a pure "gap to the car ahead" effect.
BOOLEAN = ("sc_active", "vsc_active", "red_flag_active")
CATEGORICAL = ("compound",)
TARGET = "won"

# `season` is in the feature table but deliberately not in the design matrix.
# The split is 2023-24 train, 2025-26 test, so a season term is pure
# extrapolation: a linear coefficient runs off the end of its range and a tree
# split on it just memorises which years it saw. Season stays in the table for
# grouping and breakdowns.
#
# `p_market` is excluded for the obvious reason: the model exists to be
# compared against the market, and a model fed the market's answer is not a
# comparison.
EXCLUDED = ("season", "p_market", "market_overround", "circuit_stop_prior", "total_laps")


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The columns a model is allowed to see, in a stable order."""
    missing = [c for c in (*NUMERIC, *BOOLEAN, *CATEGORICAL) if c not in df.columns]
    if missing:
        raise KeyError(f"feature frame is missing {missing}")
    out = df[list(NUMERIC)].apply(pd.to_numeric, errors="coerce")
    for col in BOOLEAN:
        out[col] = df[col].fillna(False).astype(float)
    for col in CATEGORICAL:
        out[col] = df[col].astype("object").where(df[col].notna(), None)
    return out


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
    """Linear baseline; the interpretable reference for the GBM's coefficients.

    No class weighting. The positive rate is about 1 in 20 by construction and
    `class_weight="balanced"` would fix that at the cost of wrecking
    calibration -- which is the only thing this project measures.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2_000) -> None:
        self.C = C
        self.max_iter = max_iter
        self.pipeline_: Pipeline | None = None

    def _build(self) -> Pipeline:
        numeric = Pipeline(
            [
                # add_indicator keeps "this gap was unmeasurable" as a signal.
                # A lapped car's gap_to_leader_s is null, and median-imputing it
                # to a mid-field gap would say something false about the car.
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        pre = ColumnTransformer(
            [
                ("num", numeric, list(NUMERIC)),
                ("bool", "passthrough", list(BOOLEAN)),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                    list(CATEGORICAL),
                ),
            ]
        )
        return Pipeline(
            [("pre", pre), ("clf", LogisticRegression(C=self.C, max_iter=self.max_iter))]
        )

    def fit(self, df: pd.DataFrame) -> LogisticModel:
        self.pipeline_ = self._build()
        self.pipeline_.fit(design_matrix(df), df[TARGET].astype(int))
        return self

    def predict_raw(self, df: pd.DataFrame) -> pd.Series:
        if self.pipeline_ is None:
            raise RuntimeError("fit() before predict")
        p = self.pipeline_.predict_proba(design_matrix(df))[:, 1]
        return pd.Series(p, index=df.index, dtype="float64")

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return normalise_by_lap(df.assign(p_raw=self.predict_raw(df)))

    def coefficients(self) -> pd.Series:
        """Fitted coefficients by feature name, largest absolute value first.

        Read the signs before trusting anything downstream: position and
        gap_to_leader_s must come out negative, lap_fraction interacts with
        them, and a sign that reads backwards means the design matrix is wrong,
        not that F1 is surprising.
        """
        if self.pipeline_ is None:
            raise RuntimeError("fit() before reading coefficients")
        names = self.pipeline_.named_steps["pre"].get_feature_names_out()
        coef = self.pipeline_.named_steps["clf"].coef_[0]
        s = pd.Series(coef, index=[n.split("__", 1)[-1] for n in names])
        return s.reindex(s.abs().sort_values(ascending=False).index)


# Shallow and boring on purpose. ~93k rows but only 84 independent races, and
# the effective sample size is races, not rows: every lap of a race shares one
# outcome. Depth here buys memorised race shapes, not signal.
GBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


class GBMModel:
    """LightGBM binary classifier, grouped by race for any CV.

    Nulls are handed to LightGBM as nulls rather than imputed: it learns a
    default direction per split, which is strictly more informative than the
    median substitution the logistic model needs.
    """

    def __init__(self, **params) -> None:
        self.params = {**GBM_PARAMS, **params}
        self.model_: lgb.LGBMClassifier | None = None

    @staticmethod
    def _frame(df: pd.DataFrame) -> pd.DataFrame:
        x = design_matrix(df)
        for col in CATEGORICAL:
            x[col] = x[col].astype("category")
        return x

    def fit(self, df: pd.DataFrame) -> GBMModel:
        self.model_ = lgb.LGBMClassifier(**self.params)
        self.model_.fit(self._frame(df), df[TARGET].astype(int))
        return self

    def predict_raw(self, df: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("fit() before predict")
        p = self.model_.predict_proba(self._frame(df))[:, 1]
        return pd.Series(p, index=df.index, dtype="float64")

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return normalise_by_lap(df.assign(p_raw=self.predict_raw(df)))

    def importances(self) -> pd.Series:
        """Gain-based importance, largest first.

        Compare the top of this against LogisticModel.coefficients(). A feature
        the GBM leans on that the linear model gives the opposite sign to means
        one of them is fitting noise, and that is worth resolving before any
        number goes in a results table.
        """
        if self.model_ is None:
            raise RuntimeError("fit() before reading importances")
        booster = self.model_.booster_
        s = pd.Series(
            booster.feature_importance(importance_type="gain"),
            index=booster.feature_name(),
            dtype="float64",
        )
        return (s / s.sum()).sort_values(ascending=False)

    def save(self, path) -> None:
        if self.model_ is None:
            raise RuntimeError("fit() before save")
        joblib.dump(self.model_, Path(path))

    @classmethod
    def load(cls, path) -> GBMModel:
        obj = cls()
        obj.model_ = joblib.load(Path(path))
        return obj
