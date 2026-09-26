"""Eval-side invariants. Nothing here touches DuckDB or the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairlap.eval import calibration, metrics
from fairlap.eval import compare_market as cm


def paired_fixture(n_races: int = 6, n_laps: int = 30, n_drivers: int = 4) -> pd.DataFrame:
    """A paired-frame-shaped table where the market is sharper than the model.

    The leader wins every race, the market knows it and the model is vague, so
    every comparison here has a known direction.
    """
    rng = np.random.default_rng(0)
    rows = []
    for race in range(n_races):
        for lap in range(1, n_laps + 1):
            for pos in range(1, n_drivers + 1):
                leader = pos == 1
                rows.append(
                    {
                        "session_key": 100 + race,
                        "lap_number": lap,
                        "driver_number": pos,
                        "season": 2025 + race % 2,
                        "total_laps": n_laps,
                        "position": pos,
                        "sc_active": lap in (12, 13),
                        "vsc_active": False,
                        "red_flag_active": False,
                        "market_overround": 1.05,
                        "p_market": 0.9 if leader else 0.1 / (n_drivers - 1),
                        "p_gbm": 0.6 if leader else 0.4 / (n_drivers - 1),
                        "p_frozen_prerace": 0.5 if leader else 0.5 / (n_drivers - 1),
                        "won": int(leader),
                        "noise": rng.random(),
                    }
                )
    return pd.DataFrame(rows)


MODELS = ("gbm", "frozen_prerace")


def test_brier_and_log_loss_on_known_numbers():
    y = pd.Series([1, 0])
    p = pd.Series([0.75, 0.25])
    assert metrics.brier(y, p) == pytest.approx(0.0625)
    assert metrics.log_loss(y, p) == pytest.approx(-np.log(0.75))


def test_a_null_prediction_is_dropped_not_imputed():
    """A model with no opinion scores nothing, rather than scoring 0.5."""
    y = pd.Series([1, 0, 1])
    p = pd.Series([0.75, 0.25, np.nan])
    assert metrics.brier(y, p) == pytest.approx(0.0625)
    assert np.isnan(metrics.brier(pd.Series([1]), pd.Series([np.nan])))


def test_phase_boundaries():
    df = pd.DataFrame({"lap_number": [1, 10, 11, 40, 41, 50], "total_laps": [50] * 6})
    assert list(metrics.label_phase(df)) == [
        "opening",
        "opening",
        "middle",
        "middle",
        "closing",
        "closing",
    ]


def test_a_short_race_resolves_the_phase_overlap_toward_opening():
    """Lap 3 of a red-flagged 15-lap race is an opening lap, not a closing one."""
    df = pd.DataFrame({"lap_number": [3, 12], "total_laps": [15, 15]})
    assert list(metrics.label_phase(df)) == ["opening", "closing"]


def test_condition_pools_sc_vsc_and_red_flag():
    df = pd.DataFrame(
        {
            "sc_active": [False, True, False, False],
            "vsc_active": [False, False, True, False],
            "red_flag_active": [False, False, False, True],
        }
    )
    assert list(metrics.label_condition(df)) == [
        "green",
        "interrupted",
        "interrupted",
        "interrupted",
    ]


def test_score_by_reports_races_beside_rows():
    df = metrics.with_labels(paired_fixture())
    out = metrics.score_by(df, ["phase"], ["p_market", "p_gbm"])
    assert set(out["phase"]) == {"opening", "middle", "closing"}
    assert (out["races"] == 6).all()
    assert (out["p_market_brier"] < out["p_gbm_brier"]).all()


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate():
    df = metrics.with_labels(paired_fixture())
    stat = metrics.brier_diff("p_gbm")
    point = stat(df)
    first = metrics.bootstrap_ci(df, stat, n_boot=300, seed=7)
    again = metrics.bootstrap_ci(df, stat, n_boot=300, seed=7)
    assert first == again
    assert first[0] <= point <= first[1]


def test_quantile_bins_are_equal_count_and_uniform_bins_are_not():
    """Why the calibration curve uses quantile bins: the probabilities pile up."""
    df = pd.DataFrame(
        {"p": np.concatenate([np.full(900, 0.01) + np.arange(900) / 1e5, np.linspace(0.3, 1, 100)])}
    )
    df["won"] = (df["p"] > 0.5).astype(int)
    quantile = calibration.calibration_table(df, ["p"], n_bins=10, strategy="quantile")
    uniform = calibration.calibration_table(df, ["p"], n_bins=10, strategy="uniform")
    assert quantile["n"].max() - quantile["n"].min() <= 1
    assert uniform["n"].max() > 50 * uniform["n"].min()


def test_calibration_table_rejects_an_unknown_strategy():
    df = pd.DataFrame({"p": [0.1, 0.9], "won": [0, 1]})
    with pytest.raises(ValueError, match="strategy must be"):
        calibration.calibration_table(df, ["p"], n_bins=2, strategy="equal-width")


def test_expected_calibration_error_is_weighted_by_bin_count():
    table = pd.DataFrame({"pred": ["p", "p"], "bin": [0, 1], "n": [90, 10], "gap": [0.0, 0.5]})
    assert calibration.expected_calibration_error(table)["p"] == pytest.approx(0.05)


def test_paired_frame_drops_a_row_either_side_cannot_price():
    df = paired_fixture(n_races=2, n_laps=2)
    df.loc[0, "p_market"] = np.nan
    df.loc[1, "p_gbm"] = np.nan
    df.loc[2, "p_frozen_prerace"] = np.nan
    out = cm.paired_frame(df, MODELS)
    assert len(out) == len(df) - 3
    assert out[["p_market", "p_gbm", "p_frozen_prerace"]].notna().all().all()


def test_paired_frame_puts_model_and_market_on_the_same_support():
    """Both sides renormalised over the retained drivers, and only over those.

    Without this the market sums to 1 over the drivers it priced while the
    model sums to less than 1 over the drivers that survived the join, and the
    gap is scored as if it were forecasting skill.
    """
    df = paired_fixture(n_races=2, n_laps=2)
    df.loc[0, "p_market"] = np.nan
    out = cm.paired_frame(df, MODELS)
    for col in ("p_market", "p_gbm", "p_frozen_prerace"):
        sums = out.groupby(["session_key", "lap_number"])[col].sum()
        assert np.allclose(sums, 1.0)
    # The pre-scaling numbers survive, so a result can be quoted either way.
    assert (out["p_gbm_raw"] <= out["p_gbm"] + 1e-12).all()


def test_paired_comparison_reads_the_ci_not_the_point_estimate():
    df = cm.paired_frame(paired_fixture(), MODELS)
    out = cm.paired_comparison(df, None, MODELS, n_boot=300)
    assert set(out["model"]) == set(MODELS)
    assert (out["brier_diff"] > 0).all()
    assert (out["verdict"] == "market").all()
    assert (out["ci_lo"] <= out["brier_diff"]).all()
    assert (out["races"] == 6).all()


def test_a_tie_is_reported_as_a_tie():
    df = cm.paired_frame(paired_fixture(), MODELS)
    # Hand the "model" the market's own numbers: the difference is exactly zero
    # and no amount of resampling may turn that into a win.
    df["p_gbm"] = df["p_market"]
    out = cm.paired_comparison(df, None, ["gbm"], n_boot=300)
    assert out.loc[0, "brier_diff"] == pytest.approx(0.0)
    assert out.loc[0, "verdict"] == "tie"


def test_the_backtest_pays_the_quoted_price_not_the_devigged_one():
    """De-vigging removes the house edge; transacting at it invents PnL."""
    df = cm.paired_frame(paired_fixture(n_races=2, n_laps=2), MODELS)
    bets = cm.betting_backtest(df, "gbm", edge_threshold=0.01, spread=0.0)
    assert not bets.empty

    quoted = (df["p_market_raw"] * df["market_overround"]).clip(upper=1.0)
    edge = df["p_gbm"] - df["p_market"]
    expected = np.where(edge > 0.01, quoted, 1 - quoted)[edge.abs() > 0.01].sum()
    assert bets.loc[0, "staked"] == pytest.approx(expected)

    # And the de-vigged basis would have staked something else: the overround
    # is exactly the house edge, so pricing off it is a discount nobody offered.
    devigged = np.where(edge > 0.01, df["p_market"], 1 - df["p_market"])[edge.abs() > 0.01].sum()
    assert expected != pytest.approx(devigged)


def test_the_backtest_makes_money_only_when_the_model_knows_something():
    df = cm.paired_frame(paired_fixture(), MODELS)
    df["p_gbm"] = df["won"].astype(float)
    good = cm.betting_backtest(df, "gbm", spread=0.0)
    assert good.loc[0, "pnl"] > 0

    df["p_gbm"] = 1 - df["won"].astype(float)
    bad = cm.betting_backtest(df, "gbm", spread=0.0)
    assert bad.loc[0, "pnl"] < 0


def test_the_headline_names_a_group_and_a_direction():
    df = cm.paired_frame(paired_fixture(), MODELS)
    out = cm.paired_comparison(df, ["phase"], ["gbm"], n_boot=300)
    line = cm.headline(out, "gbm")
    assert "loses to the market" in line
    assert "CI" in line and "races" in line
