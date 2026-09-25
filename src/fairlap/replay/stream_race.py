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
from fairlap.transform.build_features import build_race, lap_frame, stop_count_prior
from fairlap.transform.race_inputs import RaceInputs, load


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


def run(session_key: int, model_name: str = "gbm", speed: float | None = None) -> pd.DataFrame:
    """Replay a race, predict at each lap, write to the `predictions` table."""
    raise NotImplementedError(
        "run() needs a fitted model, which lands in Phase 4. stream() is complete "
        "and is what Phase 2's leakage test exercises."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a race lap by lap")
    parser.add_argument("--session-key", type=int, required=True)
    parser.add_argument("--model", default=None)
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
