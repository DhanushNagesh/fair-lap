"""Write the predictions table: one probability per (race, lap, driver, model).

Every model here is scored out of fold. A race is either entirely in the
training half of a fold or entirely in the held-out half -- `GroupKFold` on
`session_key`, never `KFold`, because laps from one race share an outcome and
splitting them across the fold boundary reports a fantasy score.

Test seasons are not touched. Phase 3 produces out-of-fold predictions on the
training seasons only; refitting on all of train and scoring test is Phase 4's
job and happens once. `--allow-test` exists so that step has a way in, and
nothing else should pass it.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import GroupKFold

from fairlap import db
from fairlap.config import TEST_SEASONS, TRAIN_SEASONS
from fairlap.model.baseline import FrozenPreRaceBaseline, PositionRateBaseline
from fairlap.model.gbm import GBMModel, LogisticModel

KEY = ("session_key", "lap_number", "driver_number", "model")

MODELS = {
    "position_rate": PositionRateBaseline,
    "frozen_prerace": FrozenPreRaceBaseline,
    "logistic": LogisticModel,
    "gbm": GBMModel,
}

# The frozen market baseline learns nothing across races -- its anchor is lap-1
# data of the race being predicted -- so folding it would cost time and change
# nothing.
UNFITTED = {"frozen_prerace"}

_DDL = """
    session_key   BIGINT,
    lap_number    BIGINT,
    driver_number BIGINT,
    model         VARCHAR,
    p             DOUBLE,
    fold          INTEGER
"""


def init_schema(con) -> None:
    con.execute(f"CREATE TABLE IF NOT EXISTS predictions ({_DDL})")


def out_of_fold(name: str, df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold probabilities for one model over `df`, grouped by race."""
    factory = MODELS[name]
    out = pd.Series(index=df.index, dtype="float64")
    fold_id = pd.Series(index=df.index, dtype="float64")

    if name in UNFITTED:
        out.loc[:] = factory().predict_proba(df)
        fold_id.loc[:] = -1
    else:
        races = df["session_key"].nunique()
        splits = min(n_splits, races)
        if splits < 2:
            raise ValueError(f"{races} race(s) is not enough to fold; need at least 2")
        splitter = GroupKFold(n_splits=splits)
        for i, (train_idx, test_idx) in enumerate(
            splitter.split(df, groups=df["session_key"].to_numpy())
        ):
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            # Normalisation happens inside predict_proba, and it has to see the
            # whole held-out fold: a lap's denominator is the field on that lap,
            # which only exists if the fold keeps races intact. It does.
            out.iloc[test_idx] = factory().fit(train).predict_proba(test).to_numpy()
            fold_id.iloc[test_idx] = i

    return pd.DataFrame(
        {
            "session_key": df["session_key"].astype("int64"),
            "lap_number": df["lap_number"].astype("int64"),
            "driver_number": df["driver_number"].astype("int64"),
            "model": name,
            "p": out.to_numpy(),
            "fold": fold_id.astype("Int64").to_numpy(),
        }
    )


def run(
    seasons=TRAIN_SEASONS,
    models: list[str] | None = None,
    n_splits: int = 5,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Fit, predict out of fold and persist. Returns the written frame."""
    seasons = tuple(seasons)
    leaked = sorted(set(seasons) & set(TEST_SEASONS))
    if leaked and not allow_test:
        raise ValueError(
            f"{leaked} are test seasons and Phase 3 does not score them. "
            "Pass allow_test=True only for the single final evaluation."
        )

    names = models or list(MODELS)
    con = db.connect()
    try:
        init_schema(con)
        df = con.execute(
            "SELECT * FROM features WHERE season IN (SELECT UNNEST($1::BIGINT[])) "
            "ORDER BY session_key, lap_number, driver_number",
            [list(seasons)],
        ).fetchdf()
        if df.empty:
            raise ValueError(f"no feature rows for seasons {seasons}; run `make features`")

        frames = [out_of_fold(name, df, n_splits=n_splits) for name in names]
        out = pd.concat(frames, ignore_index=True)
        db.upsert(con, "predictions", out, key=KEY)
        return out
    finally:
        con.close()


def summary(out: pd.DataFrame) -> pd.DataFrame:
    """Per-model row counts and the worst per-lap sum, which must be 1 or null."""
    rows = []
    for name, part in out.groupby("model"):
        scored = part.dropna(subset=["p"])
        sums = scored.groupby(["session_key", "lap_number"])["p"].sum()
        rows.append(
            {
                "model": name,
                "rows": len(part),
                "scored": len(scored),
                "laps": len(sums),
                "min_lap_sum": sums.min() if len(sums) else float("nan"),
                "max_lap_sum": sums.max() if len(sums) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write out-of-fold model predictions")
    parser.add_argument("--seasons", type=int, nargs="*", default=list(TRAIN_SEASONS))
    parser.add_argument("--models", nargs="*", default=None, choices=list(MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="score the test seasons; Phase 4 only, and only once",
    )
    args = parser.parse_args()
    out = run(args.seasons, args.models, n_splits=args.folds, allow_test=args.allow_test)
    print(summary(out).to_string(index=False))
