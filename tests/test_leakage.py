"""Leakage tests. These are the tests that make the project's result mean anything.

Each one encodes a rule from the scope. None of them may be weakened to make a
model look better.

The load-bearing one is `test_replay_matches_full_build`. Everything else here
checks a single rule in isolation; that one checks the whole pipeline at once,
by rebuilding each lap from history truncated at that lap's end and demanding
the same numbers. A feature that reaches forward cannot pass it, whether or not
anybody thought to write a test for that particular feature.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from fairlap.config import TEST_SEASONS, TRAIN_SEASONS
from fairlap.transform.asof import asof_join_price
from fairlap.transform.build_features import FEATURE_COLUMNS, build_race, lap_frame

PRIOR = pd.Series({7: 1.5})
GLOBAL_PRIOR = 1.5
COMPARED = [
    "session_key",
    "lap_number",
    "driver_number",
    *FEATURE_COLUMNS,
    "p_market",
    "p_market_prerace",
]


def _indexed(df: pd.DataFrame) -> pd.DataFrame:
    return df[COMPARED].set_index(COMPARED[:3]).sort_index()


def _differences(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int]:
    """Columns where the two frames disagree, ignoring null == null."""
    out: dict[str, int] = {}
    for col in left.columns:
        a, b = left[col], right[col]
        numeric = (
            pd.api.types.is_numeric_dtype(a)
            and pd.api.types.is_numeric_dtype(b)
            and not pd.api.types.is_bool_dtype(a)
            and not pd.api.types.is_bool_dtype(b)
        )
        if numeric:
            differs = ~((a - b).abs() <= 1e-9)
            differs &= ~(a.isna() & b.isna())
        else:
            # Elementwise: `!=` on an object column holding pd.NA raises
            # rather than returning False.
            differs = pd.Series(
                [
                    not (pd.isna(x) and pd.isna(y))
                    and not (pd.isna(x) or pd.isna(y))
                    and x != y
                    or (pd.isna(x) != pd.isna(y))
                    for x, y in zip(a, b, strict=True)
                ],
                index=a.index,
            )
        if differs.any():
            out[col] = int(differs.sum())
    return out


def test_replay_matches_full_build(synthetic_race):
    """Rule 1, end to end: a lap rebuilt from truncated history is the same lap.

    Truncation is per driver and strict -- everything after *that* driver's own
    lap end is removed, not merely everything after the field's. Any feature
    reading a later timestamp changes value here and the test fails.
    """
    full = _indexed(build_race(synthetic_race, PRIOR, GLOBAL_PRIOR))
    grid = lap_frame(synthetic_race)

    checked = 0
    for row in grid.itertuples(index=False):
        if pd.isna(row.lap_end):
            continue
        truncated = synthetic_race.truncate(int(row.lap_number), row.lap_end)
        rebuilt = build_race(truncated, PRIOR, GLOBAL_PRIOR, with_target=False)
        one = rebuilt[
            (rebuilt["lap_number"] == row.lap_number)
            & (rebuilt["driver_number"] == row.driver_number)
        ]
        assert len(one) == 1, f"lap {row.lap_number} driver {row.driver_number} vanished"

        key = (row.session_key, row.lap_number, row.driver_number)
        diff = _differences(full.loc[[key]], _indexed(one))
        assert not diff, f"{key} differs when rebuilt from truncated history: {diff}"
        checked += 1

    assert checked == len(grid)


def test_stream_matches_full_build(synthetic_race, monkeypatch):
    """The streamer Phase 5 consumes agrees with the table Phase 3 trains on."""
    from fairlap.replay import stream_race

    monkeypatch.setattr(stream_race, "load", lambda *a, **k: synthetic_race)
    monkeypatch.setattr(stream_race, "stop_count_prior", lambda *a, **k: (PRIOR, GLOBAL_PRIOR))

    full = _indexed(build_race(synthetic_race, PRIOR, GLOBAL_PRIOR))
    replayed = pd.concat(
        [state.drivers for state in stream_race.stream(9999, con=object())], ignore_index=True
    )
    replayed = _indexed(replayed)

    assert full.index.equals(replayed.index)
    assert not _differences(full, replayed)


def test_the_replay_never_sees_the_target(synthetic_race):
    """`won` is the only column allowed to know the outcome, so the replay lacks it."""
    rebuilt = build_race(synthetic_race, PRIOR, GLOBAL_PRIOR, with_target=False)
    assert "won" not in rebuilt.columns
    assert "won" in build_race(synthetic_race, PRIOR, GLOBAL_PRIOR).columns


def test_expected_remaining_stops_ignores_actual_future_stops(synthetic_race):
    """Rule 3: deleting the stops a driver went on to make changes nothing at lap t.

    Driver 1 starts a second stint on lap 3. At lap 2 the feature must not know
    that, and removing the stint entirely must leave lap 2 untouched.
    """
    import dataclasses

    full = build_race(synthetic_race, PRIOR, GLOBAL_PRIOR)

    without_future = dataclasses.replace(
        synthetic_race,
        stints=synthetic_race.stints[synthetic_race.stints["lap_start"] <= 2],
        pit=synthetic_race.pit[synthetic_race.pit["lap_number"] <= 2],
    )
    trimmed = build_race(without_future, PRIOR, GLOBAL_PRIOR)

    cols = ["expected_remaining_stops", "stops_made"]
    early = full["lap_number"] <= 2
    pd.testing.assert_frame_equal(
        full.loc[early, cols].reset_index(drop=True),
        trimmed.loc[trimmed["lap_number"] <= 2, cols].reset_index(drop=True),
    )
    # And the feature does move once the stop is genuinely in the past, or the
    # test above would pass on a column that is simply constant.
    assert full.loc[full["lap_number"] == 3, "stops_made"].max() == 1


def test_prerace_anchor_ignores_every_fill_after_lights_out(synthetic_race):
    """Rule 1 for Baseline B: the closing line may not see the race it opens.

    The fixture trades at 12:00:00 (lights out) and 12:03:00 (lap 4). Neither
    is before the start, so there is no anchor at all -- and a fill added
    inside the hour before must be the one that wins, not the later ones.
    """
    from fairlap.transform.build_features import add_prerace_market, lap_frame

    grid = lap_frame(synthetic_race)
    assert add_prerace_market(grid, synthetic_race)["p_market_prerace"].isna().all()

    early = synthetic_race.market.copy()
    pre = pd.DataFrame(
        {
            "session_key": 9999,
            "driver_number": [1, 44],
            "ts": pd.to_datetime(["2025-01-01T11:30:00Z"] * 2, utc=True).astype(
                "datetime64[ns, UTC]"
            ),
            "price": [0.70, 0.30],
        }
    )
    with_pre = replace(synthetic_race, market=pd.concat([pre, early], ignore_index=True))
    anchored = add_prerace_market(lap_frame(with_pre), with_pre)
    by_driver = anchored.groupby("driver_number")["p_market_prerace"].first()
    # 0.70 / (0.70 + 0.30), not the 0.80 / 0.25 the market traded at during
    # the race, and not the 0.60 / 0.45 print at lights out.
    assert by_driver.loc[1] == pytest.approx(0.70)
    assert by_driver.loc[44] == pytest.approx(0.30)


def test_prerace_anchor_stops_at_the_window_edge(synthetic_race):
    """A fill older than PRERACE_WINDOW_MIN is not the closing line."""
    from fairlap.transform.build_features import add_prerace_market, lap_frame

    stale = pd.DataFrame(
        {
            "session_key": 9999,
            "driver_number": [1, 44],
            "ts": pd.to_datetime(["2025-01-01T10:00:00Z"] * 2, utc=True).astype(
                "datetime64[ns, UTC]"
            ),
            "price": [0.70, 0.30],
        }
    )
    aged = replace(
        synthetic_race, market=pd.concat([stale, synthetic_race.market], ignore_index=True)
    )
    assert add_prerace_market(lap_frame(aged), aged)["p_market_prerace"].isna().all()


def test_the_stint_end_lap_is_never_loaded():
    """Rule 2, structurally: a stint's final lap is hindsight, so it is not read.

    Guarding the column at the point it would enter the process is stronger
    than asking every feature not to use it.
    """
    import inspect

    from fairlap.transform import race_inputs

    source = inspect.getsource(race_inputs.load)
    stint_query = next(line for line in source.splitlines() if "raw_stints" in line)
    assert "lap_end" not in stint_query


def test_market_join_is_strictly_backward(two_lap_race, minute_prices):
    """Rule 4: lap 2 (11:02:00) must take the 11:00:00 price, never the 11:02:30 one."""
    joined = asof_join_price(two_lap_race, minute_prices, tolerance_s=300)
    lap2 = joined[(joined["lap_number"] == 2) & (joined["driver_number"] == 1)]
    assert lap2["price"].iloc[0] == pytest.approx(0.60)

    # Lap 3 ends at 11:03:00, after the 11:02:30 print, so it does take it.
    lap3 = joined[(joined["lap_number"] == 3) & (joined["driver_number"] == 1)]
    assert lap3["price"].iloc[0] == pytest.approx(0.80)


def test_a_nearest_join_would_have_leaked(two_lap_race, minute_prices):
    """The join that rule 4 forbids would give a different, later price.

    Without this, `test_market_join_is_strictly_backward` could pass simply
    because no future price existed to be pulled in.
    """
    backward = asof_join_price(two_lap_race, minute_prices, tolerance_s=300)
    nearest = pd.merge_asof(
        two_lap_race.sort_values("lap_end"),
        minute_prices.sort_values("ts"),
        left_on="lap_end",
        right_on="ts",
        by=["session_key", "driver_number"],
        direction="nearest",
    )
    lap2_backward = backward.loc[
        (backward["lap_number"] == 2) & (backward["driver_number"] == 1), "price"
    ].iloc[0]
    lap2_nearest = nearest.loc[
        (nearest["lap_number"] == 2) & (nearest["driver_number"] == 1), "price"
    ].iloc[0]
    assert lap2_nearest == pytest.approx(0.80)
    assert lap2_backward != pytest.approx(lap2_nearest)


def test_stale_price_beyond_tolerance_is_null(two_lap_race, minute_prices):
    """Rule 4: a price older than tolerance_s is dropped, not carried forward."""
    joined = asof_join_price(two_lap_race, minute_prices, tolerance_s=30)
    lap2 = joined[(joined["lap_number"] == 2) & (joined["driver_number"] == 1)]
    # The only earlier print is 120s old.
    assert pd.isna(lap2["price"].iloc[0])

    # Lap 1 ends 60s after the 11:00:00 print, so 30s is not enough either,
    # while the generous tolerance keeps it -- the null is the tolerance
    # biting, not the join failing to find anything.
    assert pd.isna(joined.loc[joined["lap_number"] == 1, "price"]).all()
    assert asof_join_price(two_lap_race, minute_prices, tolerance_s=300)["price"].notna().any()


def test_train_test_split_never_shares_a_race():
    """Rule 3: no session_key on both sides, enforced by splitting on season."""
    assert not set(TRAIN_SEASONS) & set(TEST_SEASONS)

    con = pytest.importorskip("duckdb")  # noqa: F841
    from fairlap import db
    from fairlap.transform.race_inputs import race_sessions

    try:
        con = db.connect(read_only=True)
    except Exception:
        pytest.skip("no DuckDB file; the season-level assertion above still holds")
    try:
        train = set(race_sessions(con, TRAIN_SEASONS)["session_key"])
        test = set(race_sessions(con, TEST_SEASONS)["session_key"])
    finally:
        con.close()
    assert train and test
    assert not train & test


def _real_race_or_skip(session_key: int):
    """Load one real race, or skip. The archive is gitignored, so CI has no DB."""
    from fairlap import db
    from fairlap.transform.build_features import stop_count_prior
    from fairlap.transform.race_inputs import load, market_fills

    try:
        con = db.connect(read_only=True)
    except Exception as exc:  # no DuckDB file at all
        pytest.skip(f"no local DuckDB archive: {exc}")
    try:
        con.execute("SELECT 1 FROM raw_laps LIMIT 1").fetchone()
        prior, global_prior = stop_count_prior(con)
        market = market_fills(con, [session_key])
        inputs = load(session_key, con=con, market=market)
    except Exception as exc:
        pytest.skip(f"session {session_key} not ingested: {exc}")
    finally:
        con.close()
    return inputs, prior, global_prior


# Melbourne 2025 is the regression: a red-flagged race where OpenF1 emits
# several stints sharing one lap_start, which used to leave the as-of join to
# break the tie on frame order and put the replay and the table on different
# tyres. Suzuka 2025 covers the market panel, where the overround column used
# to be populated for drivers who only traded later in the race.
@pytest.mark.parametrize("session_key", [9693, 10006, 9904])
def test_replay_matches_full_build_on_a_real_race(session_key):
    """Rule 1 against real data, which is messier than any fixture."""
    inputs, prior, global_prior = _real_race_or_skip(session_key)

    full = _indexed(build_race(inputs, prior, global_prior))
    grid = lap_frame(inputs)
    cutoffs = grid["lap_end"].fillna(grid["lap_start"]).groupby(grid["lap_number"]).max()

    rebuilt = []
    for lap_number, cutoff in cutoffs.items():
        if pd.isna(cutoff):
            continue
        state = build_race(
            inputs.truncate(int(lap_number), cutoff), prior, global_prior, with_target=False
        )
        rebuilt.append(state[state["lap_number"] == lap_number])

    replayed = _indexed(pd.concat(rebuilt, ignore_index=True))
    assert full.index.equals(replayed.index)
    assert not _differences(full, replayed)


def test_the_bootstrap_resamples_whole_races():
    """Rule 6: a resample draws races, never rows.

    Checked by having the statistic report what it was handed: every race in a
    resample must appear with its full lap count (or a multiple of it, when the
    same race is drawn twice). A row-level bootstrap fails this immediately.
    """
    from fairlap.eval.metrics import bootstrap_ci

    df = pd.DataFrame(
        {
            "session_key": [1] * 5 + [2] * 3 + [3] * 7,
            "won": [0] * 15,
        }
    )
    sizes = {1: 5, 2: 3, 3: 7}
    seen = []

    def stat(part):
        counts = part["session_key"].value_counts()
        seen.append({int(k): int(v) for k, v in counts.items()})
        return float(len(part))

    bootstrap_ci(df, stat, n_boot=50, seed=1)
    assert seen
    for counts in seen:
        assert sum(counts.values()) == sum(sizes[k] * (v // sizes[k]) for k, v in counts.items())
        for key, n in counts.items():
            assert n % sizes[key] == 0, f"race {key} was split: {n} rows"


def test_a_row_bootstrap_would_have_manufactured_significance():
    """Rule 6, stated as the failure it prevents.

    Three races that are internally unanimous and disagree with each other.
    The race-clustered interval has to be wide enough to admit that three
    observations say very little; a row-level interval on the same 150 rows
    reports a tenth of the width and would call this significant.
    """
    from fairlap.eval.metrics import bootstrap_ci

    df = pd.DataFrame(
        {
            "session_key": [1] * 50 + [2] * 50 + [3] * 50,
            "won": [0.0] * 100 + [1.0] * 50,
            "row_id": range(150),
        }
    )

    def mean_won(part):
        return float(part["won"].mean())

    clustered = bootstrap_ci(df, mean_won, n_boot=2_000, seed=0)
    by_row = bootstrap_ci(df, mean_won, n_boot=2_000, cluster="row_id", seed=0)
    # Clustered: 0 to 1, because three races can all come up the same way.
    # Row-level: about 0.26 to 0.41, a seventh of the width, from the same data.
    assert clustered == (0.0, 1.0)
    assert (clustered[1] - clustered[0]) > 5 * (by_row[1] - by_row[0])


def test_the_holdout_pass_refuses_an_overlapping_split():
    """Rule 3 at the point where the test seasons are finally scored."""
    from fairlap.model.predict import holdout

    with pytest.raises(ValueError, match="both halves"):
        holdout(train_seasons=(2023, 2025), test_seasons=(2025, 2026))
