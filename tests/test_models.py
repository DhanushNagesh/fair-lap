"""Model-side invariants. Nothing here touches DuckDB or the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairlap.config import TEST_SEASONS
from fairlap.model import predict
from fairlap.model.baseline import FrozenPreRaceBaseline, PositionRateBaseline
from fairlap.model.gbm import CATEGORICAL, GBMModel, LogisticModel, design_matrix, normalise_by_lap

RNG = np.random.default_rng(0)


def synthetic_features(n_races: int = 12, n_laps: int = 20, n_drivers: int = 8) -> pd.DataFrame:
    """A feature-table-shaped frame where the leader usually wins.

    Enough structure that a model has something to find, and enough races that
    GroupKFold has folds to make.
    """
    rows = []
    for race in range(n_races):
        order = RNG.permutation(np.arange(1, n_drivers + 1))
        winner = int(order[0])
        for lap in range(1, n_laps + 1):
            for i, driver in enumerate(order, start=1):
                rows.append(
                    {
                        "session_key": 9000 + race,
                        "lap_number": lap,
                        "driver_number": int(driver),
                        "season": 2023 + race % 2,
                        "total_laps": n_laps,
                        "lap_fraction": lap / n_laps,
                        "laps_remaining": n_laps - lap,
                        "position": i,
                        "grid_position": i,
                        "gap_to_leader_s": 0.0 if i == 1 else float(i) * 2.0,
                        "gap_to_ahead_s": 0.0 if i == 1 else 2.0,
                        "lap_time_s": 90.0 + i * 0.1,
                        "pace_roll_s": 90.0 + i * 0.1,
                        "tyre_age_laps": float(lap % 15),
                        "stint_number": 1.0,
                        "stops_made": lap // 15,
                        "expected_remaining_stops": 1.0,
                        "laps_since_sc": float(lap - 5) if lap > 5 else np.nan,
                        "sc_active": False,
                        "vsc_active": False,
                        "red_flag_active": False,
                        "compound": "HARD" if lap < 15 else "MEDIUM",
                        "p_market": 0.5 if (lap == 1 and i <= 2) else np.nan,
                        "market_overround": 1.1,
                        "circuit_stop_prior": 1.5,
                        "won": int(driver == winner),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return synthetic_features()


def test_normalise_sums_to_one_per_lap():
    df = pd.DataFrame(
        {
            "session_key": [1, 1, 1, 2, 2],
            "lap_number": [1, 1, 2, 1, 1],
            "p_raw": [0.2, 0.6, 0.4, 0.1, 0.1],
        }
    )
    out = normalise_by_lap(df)
    sums = out.groupby([df["session_key"], df["lap_number"]]).sum()
    assert np.allclose(sums, 1.0)


def test_normalise_leaves_nulls_out_of_the_denominator():
    df = pd.DataFrame(
        {"session_key": [1, 1, 1], "lap_number": [1, 1, 1], "p_raw": [0.3, 0.1, np.nan]}
    )
    out = normalise_by_lap(df)
    assert out.isna().sum() == 1
    assert out.sum() == pytest.approx(1.0)


def test_normalise_does_not_divide_by_zero():
    df = pd.DataFrame({"session_key": [1, 1], "lap_number": [1, 1], "p_raw": [0.0, 0.0]})
    assert normalise_by_lap(df).isna().all()


def test_design_matrix_cannot_see_the_market_or_the_answer(features):
    x = design_matrix(features)
    for banned in ("p_market", "market_overround", "won", "season", "session_key"):
        assert banned not in x.columns
    assert set(CATEGORICAL) <= set(x.columns)


def test_position_baseline_ranks_the_leader_first(features):
    model = PositionRateBaseline().fit(features)
    late = features[features["lap_fraction"] > 0.8]
    p = model.predict_proba(late)
    by_position = p.groupby(late["position"]).mean()
    assert by_position.loc[1] > by_position.loc[4]
    assert np.allclose(p.groupby([late["session_key"], late["lap_number"]]).sum(), 1.0)


def test_position_baseline_never_returns_a_bare_zero_or_one(features):
    raw = PositionRateBaseline().fit(features).predict_raw(features)
    assert raw.between(0.0, 1.0, inclusive="neither").all()


def test_frozen_baseline_is_flat_across_the_race(features):
    p = FrozenPreRaceBaseline().predict_proba(features)
    priced = features.assign(p=p).dropna(subset=["p"])
    spread = priced.groupby(["session_key", "driver_number"])["p"].nunique()
    assert (spread == 1).all()


def test_frozen_baseline_ignores_prices_from_later_laps(features):
    """The anchor is lap 1. A price appearing at lap 10 must not be picked up.

    This is the leakage guard on Baseline B: reaching forward for the first
    non-null price would hand the frozen baseline a mid-race quote and quietly
    turn it into a live market model.
    """
    later = features.copy()
    mask = (later["lap_number"] == 10) & (later["driver_number"] == later["driver_number"].max())
    later.loc[mask, "p_market"] = 0.99
    baseline = FrozenPreRaceBaseline().predict_raw(later)
    # The lap-10 quote must not appear anywhere, for that driver or any other.
    assert not (baseline == 0.99).any()
    # And a (race, driver) with no lap-1 price stays unpriced rather than
    # inheriting one from further into the race.
    anchored = set(
        map(
            tuple,
            later.loc[
                (later["lap_number"] == 1) & later["p_market"].notna(),
                ["session_key", "driver_number"],
            ].to_numpy(),
        )
    )
    has_anchor = [
        (sk, dn) in anchored
        for sk, dn in zip(later["session_key"], later["driver_number"], strict=True)
    ]
    assert baseline[~np.array(has_anchor)].isna().all()
    assert baseline[np.array(has_anchor)].notna().all()


def test_models_fit_and_normalise(features):
    for model in (LogisticModel(), GBMModel(n_estimators=20)):
        p = model.fit(features).predict_proba(features)
        sums = p.groupby([features["session_key"], features["lap_number"]]).sum()
        assert np.allclose(sums, 1.0)


def test_logistic_coefficients_point_the_right_way(features):
    coef = LogisticModel().fit(features).coefficients()
    assert coef["position"] < 0
    assert coef["gap_to_leader_s"] < 0


def test_out_of_fold_keeps_each_race_whole(features):
    out = predict.out_of_fold("logistic", features, n_splits=4)
    folds_per_race = out.groupby("session_key")["fold"].nunique()
    assert (folds_per_race == 1).all()
    assert out["fold"].nunique() == 4
    assert out["p"].notna().all()


def test_out_of_fold_scores_every_row_exactly_once(features):
    out = predict.out_of_fold("gbm", features, n_splits=3)
    assert len(out) == len(features)
    assert not out.duplicated(subset=["session_key", "lap_number", "driver_number"]).any()


def test_run_refuses_the_test_seasons():
    with pytest.raises(ValueError, match="test seasons"):
        predict.run(seasons=TEST_SEASONS)
