"""Phase 5: the replay reproduces the predictions the headline was scored on."""

from __future__ import annotations

import pytest

from fairlap.model.predict import HOLDOUT_FOLD


def _db_or_skip():
    from fairlap import db

    try:
        con = db.connect(read_only=True)
        con.execute("SELECT 1 FROM predictions LIMIT 1").fetchone()
    except Exception as exc:
        pytest.skip(f"no local DuckDB archive with predictions: {exc}")
    return con


# Melbourne 2025 is a test-season race, so the replay's fit is the holdout fit.
@pytest.mark.parametrize("model_name", ["gbm", "frozen_prerace"])
def test_replay_reproduces_the_holdout_predictions(model_name):
    from fairlap.replay.stream_race import replay

    con = _db_or_skip()
    try:
        scored = con.execute(
            "SELECT lap_number, driver_number, p FROM predictions "
            "WHERE session_key = 9693 AND model = ? AND fold = ?",
            [model_name, HOLDOUT_FOLD],
        ).fetchdf()
    finally:
        con.close()
    if scored.empty:
        pytest.skip("no holdout predictions; run `make predict-test`")

    replayed = replay(9693, model_name, verbose=False)
    merged = scored.merge(replayed, on=["lap_number", "driver_number"], suffixes=("", "_r"))
    assert len(merged) == len(scored) == len(replayed)
    assert (merged["p"] - merged["p_r"]).abs().max(skipna=True) < 1e-9
    assert merged["p"].isna().equals(merged["p_r"].isna())


def test_a_training_race_is_left_out_of_its_own_fit(monkeypatch):
    """Rule 3 in the replay: a race is never predicted by a model that saw it."""
    from fairlap.replay import stream_race

    seen = {}

    class Spy:
        def fit(self, df):
            seen["races"] = set(df["session_key"])
            return self

    con = _db_or_skip()
    try:
        race = con.execute(
            "SELECT session_key FROM features WHERE season = 2024 LIMIT 1"
        ).fetchone()[0]
        monkeypatch.setitem(stream_race.MODELS, "gbm", Spy)
        stream_race.fit_for_replay(race, "gbm", con)
    finally:
        con.close()
    assert race not in seen["races"]
    assert len(seen["races"]) > 10
