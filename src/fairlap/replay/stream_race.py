"""Stream a past race one lap at a time.

The replay is the honesty mechanism: it hands the model a state object built
only from laps <= t, so a feature that needs the future cannot be computed at
all rather than quietly computing a leaked value. If `build_features` and the
replay ever disagree on a row, the replay is right.

It does that by running the *same* functions as `build_features`, over a
`RaceInputs` truncated to lap t. Nothing here re-implements a feature, because
a second implementation would only prove the two implementations agree, not
that either one respects the timestamp.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from fairlap import db
from fairlap.config import TRAIN_SEASONS
from fairlap.model.predict import MODELS, UNFITTED
from fairlap.transform.build_features import build_race, lap_frame, stop_count_prior
from fairlap.transform.race_inputs import RaceInputs, load, market_fills

KEY = ("session_key", "lap_number", "driver_number", "model")

_DDL = """
    session_key   BIGINT,
    lap_number    BIGINT,
    driver_number BIGINT,
    model         VARCHAR,
    p             DOUBLE,
    p_market      DOUBLE,
    position      DOUBLE,
    lap_end       TIMESTAMPTZ
"""


@dataclass(frozen=True)
class LapState:
    """Everything known at the end of one lap."""

    session_key: int
    lap_number: int
    total_laps: int
    lap_end: pd.Timestamp
    drivers: pd.DataFrame  # one row per driver, FEATURE_COLUMNS subset


def lap_cutoffs(inputs: RaceInputs) -> pd.Series:
    """The wall-clock instant each lap number is complete for the whole field.

    Drivers cross the line seconds apart, so the cutoff for lap t is the last
    driver's lap end. Cutting at the *first* driver's would starve the slower
    ones of data they genuinely had, and produce a disagreement with the full
    build that is an artefact of the cut rather than a leak. Per-driver
    correctness is enforced by the as-of joins, which never look past that
    driver's own lap_end.
    """
    grid = lap_frame(inputs)
    end = grid["lap_end"].fillna(grid["lap_start"])
    return end.groupby(grid["lap_number"]).max().sort_index()


def stream(
    session_key: int, speed: float | None = None, con=None, market: pd.DataFrame | None = None
) -> Iterator[LapState]:
    """Yield LapState for each lap in order.

    `speed` sleeps between laps to mimic real timing for the dashboard demo;
    None runs as fast as the data loads.
    """
    owned = con is None
    con = con or db.connect(read_only=True)
    try:
        inputs = load(session_key, con=con, market=market)
        prior, global_prior = stop_count_prior(con)
        cutoffs = lap_cutoffs(inputs)

        for lap_number, cutoff in cutoffs.items():
            if pd.isna(cutoff):
                continue
            truncated = inputs.truncate(int(lap_number), cutoff)
            # with_target=False is the point: the replay has no access to who
            # won, so a feature cannot borrow the answer even by accident.
            state = build_race(truncated, prior, global_prior, with_target=False)
            drivers = state[state["lap_number"] == lap_number].reset_index(drop=True)
            if speed:
                time.sleep(speed)
            yield LapState(
                session_key=inputs.session_key,
                lap_number=int(lap_number),
                total_laps=inputs.total_laps,
                lap_end=cutoff,
                drivers=drivers,
            )
    finally:
        if owned:
            con.close()


def fit_for_replay(session_key: int, model_name: str, con) -> object:
    """Fit `model_name` on the training seasons, never on the race being replayed.

    For a test-season race that is exactly the fit `predict --holdout` used, so
    the replay reproduces the scored predictions. For a training-season race it
    is leave-one-race-out: the race is dropped from the fit, because a model
    that trained on a race's laps would replay that race knowing who won.
    """
    factory = MODELS[model_name]
    model = factory()
    if model_name in UNFITTED:
        return model
    train = con.execute(
        "SELECT * FROM features WHERE season IN (SELECT UNNEST($1::BIGINT[])) "
        "AND session_key <> $2 ORDER BY session_key, lap_number, driver_number",
        [list(TRAIN_SEASONS), session_key],
    ).fetchdf()
    if train.empty:
        raise ValueError("no training rows; run `make features` first")
    return model.fit(train)


def replay(
    session_key: int, model_name: str = "gbm", speed: float | None = None, verbose: bool = True
) -> pd.DataFrame:
    """Replay a race and predict at each lap. Reads only; `run` does the write."""
    con = db.connect(read_only=True)
    try:
        model = fit_for_replay(session_key, model_name, con)
        # Passed explicitly because load() otherwise opens a second connection
        # it never closes, which blocks the write in run().
        market = market_fills(con, [session_key])
        laps = []
        for state in stream(session_key, speed=speed, con=con, market=market):
            d = state.drivers
            lap = pd.DataFrame(
                {
                    "session_key": d["session_key"].astype("int64"),
                    "lap_number": d["lap_number"].astype("int64"),
                    "driver_number": d["driver_number"].astype("int64"),
                    "model": model_name,
                    "p": model.predict_proba(d).to_numpy(),
                    "p_market": d["p_market"].to_numpy(),
                    "position": d["position"].to_numpy(),
                    "lap_end": state.lap_end,
                }
            )
            laps.append(lap)
            if verbose:
                _print_lap(state, lap)
    finally:
        con.close()
    return pd.concat(laps, ignore_index=True)


def run(session_key: int, model_name: str = "gbm", speed: float | None = None) -> pd.DataFrame:
    """Replay a race, predict at each lap, write to the `replay_predictions` table.

    Not the `predictions` table: that one holds the rows `make eval` scores, and
    it shares this table's key, so an upsert there would silently replace the
    holdout pass with whatever the replay last ran.

    Everything is written in one upsert at the end. DuckDB allows one writer or
    many readers per file, never both, so writing lap by lap would lock the
    dashboard out for the whole replay. `--speed` paces the console instead, and
    the dashboard has its own lap slider over the stored laps.
    """
    out = replay(session_key, model_name, speed)
    con = db.connect()
    try:
        con.execute(f"CREATE TABLE IF NOT EXISTS replay_predictions ({_DDL})")
        db.upsert(con, "replay_predictions", out, key=KEY)
    finally:
        con.close()
    return out


def _print_lap(state: LapState, lap: pd.DataFrame) -> None:
    top = lap.sort_values("p", ascending=False).head(3)
    cells = []
    for row in top.itertuples():
        market = "  -  " if pd.isna(row.p_market) else f"{row.p_market:.3f}"
        cells.append(f"#{row.driver_number:<2} {row.p:.3f} (mkt {market})")
    print(f"lap {state.lap_number:>3}/{state.total_laps}  " + "   ".join(cells), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a race lap by lap")
    parser.add_argument("--session-key", type=int, required=True)
    parser.add_argument("--model", default=None, choices=list(MODELS))
    parser.add_argument("--speed", type=float, default=None)
    args = parser.parse_args()

    if args.model:
        run(args.session_key, model_name=args.model, speed=args.speed)
        return
    # Without a model there is still something worth seeing: that the state
    # rebuilds lap by lap from truncated history at all.
    for state in stream(args.session_key, speed=args.speed):
        leader = state.drivers.sort_values("position").head(1)
        name = int(leader["driver_number"].iloc[0]) if not leader.empty else None
        print(
            f"lap {state.lap_number:>3}/{state.total_laps}  "
            f"{len(state.drivers):>2} cars  leader #{name}"
        )
