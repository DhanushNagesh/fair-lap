"""Build the lap-level feature table.

One row per (session_key, lap_number, driver_number), using only information
known at the end of that lap. Anything computed from a later lap -- final
classification, total stops, stint length in hindsight -- is leakage and is
tested against in tests/test_leakage.py.

Every `add_*` function takes the grid plus a `RaceInputs`, and reaches into the
raw frames only through a backward as-of join. That is what lets
`replay/stream_race` run the identical functions over truncated inputs and get
identical numbers; the equality is the test, not the docstring.

Feature groups:
  race state     position, gap_to_leader_s, gap_to_ahead_s, lap_time_s,
                 pace_roll_s, lap_fraction, laps_remaining
  tyres          compound, tyre_age_laps, stint_number, stops_made
  interruptions  sc_active, vsc_active, red_flag_active, laps_since_sc
  prior          expected_remaining_stops -- a per-circuit stop-count prior fit
                 on the training seasons only, net of stops already made. The
                 most predictive feature and the easiest leak.
  market         p_market (as-of, de-vigged), market_overround,
                 p_market_prerace -- the de-vigged closing line, constant
                 across the race
  target         won (0/1)

Known approximation: `total_laps` is taken as the number of laps the race
actually ran. That equals the scheduled distance except in a race cut short by
a red flag, where it quietly encodes that the race ended early. Fixing it needs
a scheduled-distance source OpenF1 does not expose; it is called out in the
README rather than papered over.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fairlap import db
from fairlap.config import (
    MARKET_GRID_TOLERANCE_S,
    PACE_WINDOW_LAPS,
    PIT_WINDOW_MIN_LAPS_REMAINING,
    PRERACE_WINDOW_MIN,
    STALENESS_TOLERANCE_MIN,
    TRAIN_SEASONS,
)
from fairlap.transform.asof import asof_join_price, devig
from fairlap.transform.race_inputs import RaceInputs, market_fills, race_sessions

FEATURE_COLUMNS = (
    "position",
    "gap_to_leader_s",
    "gap_to_ahead_s",
    "lap_time_s",
    "pace_roll_s",
    "lap_fraction",
    "laps_remaining",
    "compound",
    "tyre_age_laps",
    "stint_number",
    "stops_made",
    "expected_remaining_stops",
    "sc_active",
    "vsc_active",
    "red_flag_active",
    "laps_since_sc",
    "grid_position",
    "season",
)
TARGET = "won"
KEY = ("session_key", "lap_number", "driver_number")


def lap_frame(inputs: RaceInputs) -> pd.DataFrame:
    """Skeleton of (lap, driver) rows with lap_start/lap_end timestamps.

    `lap_end` is the lap's own start plus its own duration, never the next
    lap's start. The two are the same instant, but the next lap's row is not
    available to the replay at lap t, and a feature table the replay cannot
    reproduce is a feature table nobody can check.
    """
    laps = inputs.laps
    out = pd.DataFrame(
        {
            "session_key": laps["session_key"].astype("int64"),
            "lap_number": laps["lap_number"].astype("int64"),
            "driver_number": laps["driver_number"].astype("int64"),
            "lap_start": laps["date_start"],
            "lap_time_s": pd.to_numeric(laps["lap_duration"], errors="coerce"),
        }
    )
    out["lap_end"] = out["lap_start"] + pd.to_timedelta(out["lap_time_s"], unit="s")
    out["total_laps"] = inputs.total_laps
    out["season"] = inputs.year
    return out.sort_values(["lap_number", "driver_number"], ignore_index=True)


def add_race_state(df: pd.DataFrame, inputs: RaceInputs) -> pd.DataFrame:
    """position, gaps, rolling pace and lap_fraction as of the end of each lap."""
    out = df.copy()
    out["lap_fraction"] = out["lap_number"] / out["total_laps"]
    out["laps_remaining"] = out["total_laps"] - out["lap_number"]

    # Rolling pace is backward by construction: at lap t it averages laps
    # t-N+1..t, all of which are complete. min_periods=1 so early laps still
    # get a value rather than a null the model has to learn around.
    out["pace_roll_s"] = (
        out.sort_values("lap_number")
        .groupby("driver_number")["lap_time_s"]
        .transform(lambda s: s.rolling(PACE_WINDOW_LAPS, min_periods=1).mean())
    )

    position = inputs.position[["session_key", "driver_number", "date", "position"]].rename(
        columns={"date": "ts"}
    )
    # No tolerance: position is a state that persists until it changes, so the
    # last report is correct however long ago it was. Staleness only matters
    # for a price, which is a quote that goes off.
    out = asof_join_price(out, position, right_time="ts", tolerance_s=None)

    intervals = inputs.intervals[
        ["session_key", "driver_number", "date", "gap_to_leader", "interval"]
    ].rename(columns={"date": "ts"})
    intervals = intervals.assign(
        gap_to_leader_s=_gap_seconds(intervals["gap_to_leader"]),
        gap_to_ahead_s=_gap_seconds(intervals["interval"]),
    ).drop(columns=["gap_to_leader", "interval"])
    out = asof_join_price(out, intervals, right_time="ts", tolerance_s=None)

    # Grid position is the first position report of the session, which OpenF1
    # emits before the lights go out. Pre-race by definition, so it survives
    # truncation to lap 1.
    grid = (
        inputs.position.sort_values("date")
        .groupby(["session_key", "driver_number"], as_index=False)
        .first()[["session_key", "driver_number", "position"]]
        .rename(columns={"position": "grid_position"})
    )
    return out.merge(grid, on=["session_key", "driver_number"], how="left")


def _gap_seconds(series: pd.Series) -> pd.Series:
    """Parse an OpenF1 gap. "+1 LAP" is not a number of seconds, so it is null.

    Being lapped is real information, but it is not a gap in seconds and
    coercing it to one (0, or some large sentinel) would put a fabricated value
    in front of the model. Position already carries most of it.
    """
    return pd.to_numeric(series, errors="coerce")


def add_tyre_state(df: pd.DataFrame, inputs: RaceInputs) -> pd.DataFrame:
    """compound, tyre_age_laps, stint_number and stops_made.

    The current stint is the last one that *started* at or before lap t -- an
    as-of join on lap number. The stint's final lap is never read (it is not
    even loaded), so nothing here can depend on how long the stint turned out
    to be.
    """
    out = df.copy()
    stints = inputs.stints[
        [
            "session_key",
            "driver_number",
            "stint_number",
            "lap_start",
            "compound",
            "tyre_age_at_start",
        ]
    ].copy()

    if stints.empty:
        out["compound"] = pd.NA
        out["stint_number"] = pd.NA
        out["tyre_age_laps"] = np.nan
        out["stint_ordinal"] = np.nan
    else:
        stints = stints.rename(columns={"lap_start": "stint_lap_start"})
        # Rank rather than stint_number: truncation drops only later stints, so
        # the rank of the stints visible at lap t is the rank it has in the
        # full race. A gap in stint_number would not survive that.
        stints["stint_ordinal"] = stints.groupby(["session_key", "driver_number"])[
            "stint_lap_start"
        ].rank(method="first")
        stints["_on"] = stints["stint_lap_start"].astype("int64")
        # Same int32/int64 split as in asof.py: raw_stints.driver_number is
        # INTEGER, the grid's is BIGINT, and merge_asof will not bridge them.
        for col in ("session_key", "driver_number"):
            stints[col] = stints[col].astype("int64")
        left = out.assign(_on=out["lap_number"].astype("int64")).sort_values("_on")
        merged = pd.merge_asof(
            left,
            stints.sort_values("_on"),
            on="_on",
            by=["session_key", "driver_number"],
            direction="backward",
        )
        merged["tyre_age_laps"] = (
            merged["_on"] - merged["stint_lap_start"] + merged["tyre_age_at_start"]
        )
        out = merged.drop(columns=["_on", "stint_lap_start", "tyre_age_at_start"]).sort_values(
            ["lap_number", "driver_number"], ignore_index=True
        )

    # stops_made counts stints started, not rows in raw_pit. raw_pit is empty
    # for the first six races of 2023 and undercounts in nine more, so a
    # pit-derived count would report zero stops for races that plainly had
    # them. A driver on their nth stint has stopped n-1 times, and a stint's
    # start lap is known when that stint starts.
    out["stops_made"] = (out["stint_ordinal"].fillna(1) - 1).clip(lower=0).astype("int64")
    return out.drop(columns="stint_ordinal")


_SC_DEPLOY = "SAFETY CAR DEPLOYED"
_SC_END = ("SAFETY CAR IN THIS LAP", "SAFETY CAR WILL ENTER PITS")
_VSC_DEPLOY = ("VIRTUAL SAFETY CAR DEPLOYED", "VSC DEPLOYED")
_VSC_END = ("VIRTUAL SAFETY CAR ENDING", "VSC ENDING")


def add_interruptions(df: pd.DataFrame, inputs: RaceInputs) -> pd.DataFrame:
    """SC / VSC / red-flag state at each lap end, from race_control messages.

    Built as a state timeline -- each message flips a flag, and the lap takes
    the flag's value at its end via a backward as-of join. Reading the flag
    "at" a lap any other way means scanning for the next message, which is the
    future.
    """
    out = df.copy()
    rc = inputs.race_control
    flags = ["sc_active", "vsc_active", "red_flag_active"]

    if rc.empty:
        for col in flags:
            out[col] = False
        out["laps_since_sc"] = np.nan
        return out

    msg = rc["message"].fillna("").str.upper()
    flag = rc["flag"].fillna("").str.upper()
    scope = rc["scope"].fillna("").str.upper()

    # "VIRTUAL SAFETY CAR DEPLOYED" contains "SAFETY CAR DEPLOYED", so the real
    # safety car has to be matched to the exclusion of the virtual one.
    is_virtual = msg.str.contains("VIRTUAL") | msg.str.startswith("VSC")
    events = pd.DataFrame(
        {
            "session_key": rc["session_key"],
            "ts": rc["date"],
            "lap_number": rc["lap_number"],
            "sc_on": msg.str.contains(_SC_DEPLOY, regex=False) & ~is_virtual,
            "sc_off": msg.str.startswith(_SC_END),
            "vsc_on": msg.str.startswith(_VSC_DEPLOY),
            "vsc_off": msg.str.startswith(_VSC_END),
            "red_on": (flag == "RED") & (scope == "TRACK"),
            "red_off": (flag.isin(["GREEN", "CLEAR"])) & (scope == "TRACK"),
        }
    ).sort_values("ts")

    events = events[
        events[["sc_on", "sc_off", "vsc_on", "vsc_off", "red_on", "red_off"]].any(axis=1)
    ]
    if events.empty:
        for col in flags:
            out[col] = False
        out["laps_since_sc"] = np.nan
        return out

    events["sc_active"] = _latch(events["sc_on"], events["sc_off"])
    events["vsc_active"] = _latch(events["vsc_on"], events["vsc_off"])
    events["red_flag_active"] = _latch(events["red_on"], events["red_off"])
    # The lap a safety car was last deployed on, carried forward. Null until
    # the first one, which is honest: "laps since" has no value before then.
    events["last_sc_lap"] = events["lap_number"].where(events["sc_on"]).ffill()

    timeline = events[["session_key", "ts", *flags, "last_sc_lap"]]
    out = asof_join_price(out, timeline, right_time="ts", by=("session_key",), tolerance_s=None)
    for col in flags:
        out[col] = out[col].fillna(False).astype(bool)
    out["laps_since_sc"] = out["lap_number"] - out["last_sc_lap"]
    return out.drop(columns="last_sc_lap")


def _latch(on: pd.Series, off: pd.Series) -> pd.Series:
    """Boolean state that turns on at `on`, off at `off`, and holds between."""
    state = pd.Series(np.nan, index=on.index)
    state[on.to_numpy()] = 1.0
    state[off.to_numpy()] = 0.0
    return state.ffill().fillna(0.0).astype(bool)


def stop_count_prior(con, seasons=TRAIN_SEASONS) -> tuple[pd.Series, float]:
    """Mean stops per driver per race, by circuit, from the training seasons.

    Counted from stints (a driver on n stints stopped n-1 times) rather than
    from raw_pit, which OpenF1 did not publish for the first six races of 2023.

    Returns (per-circuit series, global fallback). Fit on `seasons` only --
    leakage rule 3. A circuit that never appears in training (a 2026 debut)
    falls back to the global training mean rather than borrowing from the test
    seasons, which would be exactly the leak this feature is most prone to.
    """
    rows = con.execute(
        """
        WITH per_driver AS (
            SELECT s.circuit_key, st.session_key, st.driver_number,
                   COUNT(*) - 1 AS stops
            FROM raw_stints st
            JOIN raw_sessions s USING (session_key)
            WHERE s.session_type = 'Race'
              AND s.year IN (SELECT UNNEST($1::INTEGER[]))
            GROUP BY 1, 2, 3
        )
        SELECT circuit_key, SUM(stops) AS stops, COUNT(*) AS driver_races
        FROM per_driver GROUP BY 1
        """,
        [list(seasons)],
    ).fetchdf()
    if rows.empty:
        return pd.Series(dtype="float64"), 2.0
    rows["mean_stops"] = rows["stops"] / rows["driver_races"].replace(0, np.nan)
    global_mean = float(rows["stops"].sum() / max(rows["driver_races"].sum(), 1))
    return rows.set_index("circuit_key")["mean_stops"], global_mean


def add_prior(
    df: pd.DataFrame, inputs: RaceInputs, prior: pd.Series, global_prior: float
) -> pd.DataFrame:
    """expected_remaining_stops from the circuit prior, net of stops already made.

    Not from how many stops the driver actually went on to make. The prior is a
    training-season constant for the circuit; the only per-lap inputs are stops
    already taken and laps left to take one in.
    """
    out = df.copy()
    circuit_prior = float(prior.get(inputs.circuit_key, global_prior))
    out["circuit_stop_prior"] = circuit_prior

    remaining = (circuit_prior - out["stops_made"]).clip(lower=0.0)
    # A stop needs laps left to happen in. Inside the last few laps the prior
    # is irrelevant -- nobody is stopping -- and leaving it non-zero would have
    # the feature insist on a stop that cannot occur.
    no_room = out["laps_remaining"] < PIT_WINDOW_MIN_LAPS_REMAINING
    out["expected_remaining_stops"] = remaining.where(~no_room, 0.0)
    return out


def add_market(df: pd.DataFrame, inputs: RaceInputs) -> pd.DataFrame:
    """De-vigged market price attached by backward as-of join on lap_end.

    Three steps, in this order, because the order is the rule:

    1. Carry each driver's last *fill* onto a one-minute grid, blanking it once
       it is older than STALENESS_TOLERANCE_MIN. Backward only.
    2. De-vig within each minute, across the drivers who are still priced there
       -- after the volume filter, which `market_fills` already applied.
    3. As-of join that minute panel onto lap_end.

    De-vigging before the staleness blanking would normalise against drivers
    whose price had already gone stale, which inflates the denominator with
    quotes nobody was trading.
    """
    out = df.copy()
    fills = inputs.market
    if fills.empty or out["lap_end"].notna().sum() == 0:
        out["p_market"] = np.nan
        out["market_overround"] = np.nan
        return out

    minutes = pd.date_range(
        fills["ts"].min().floor("min"), out["lap_end"].max().ceil("min"), freq="min"
    )
    drivers = fills["driver_number"].unique()
    grid = pd.MultiIndex.from_product([minutes, drivers], names=["ts", "driver_number"])
    grid = grid.to_frame(index=False)
    grid["session_key"] = inputs.session_key

    panel = asof_join_price(
        grid,
        fills,
        left_time="ts",
        right_time="ts",
        tolerance_s=STALENESS_TOLERANCE_MIN * 60,
    )
    panel = devig(panel)
    panel = panel[["session_key", "driver_number", "ts", "p_market", "overround"]].rename(
        columns={"overround": "market_overround"}
    )

    return asof_join_price(out, panel, right_time="ts", tolerance_s=MARKET_GRID_TOLERANCE_S)


def add_prerace_market(df: pd.DataFrame, inputs: RaceInputs) -> pd.DataFrame:
    """The de-vigged closing line: each driver's last fill before lights out.

    One number per driver for the whole race, which is what
    `FrozenPreRaceBaseline` freezes. Strictly before `race_start`, so it is
    knowable at lap 1 and survives truncation -- the replay reproduces it at
    every lap.

    De-vigged across the session rather than within a minute, because the
    anchors are last-fills at different instants and no single minute holds the
    whole grid. That is a weaker denominator than the in-race de-vig, which is
    why the same MIN_DEVIG_DRIVERS and OVERROUND_BAND gates still apply: a
    pre-race book assembled out of scattered fills can fail them, and when it
    does the race gets no anchor rather than a normalisation against a field we
    know is incomplete.
    """
    out = df.copy()
    fills = inputs.market
    window_start = inputs.race_start - pd.Timedelta(minutes=PRERACE_WINDOW_MIN)
    pre = fills[(fills["ts"] >= window_start) & (fills["ts"] < inputs.race_start)]
    if pre.empty:
        out["p_market_prerace"] = np.nan
        return out

    anchors = (
        pre.sort_values("ts")
        .groupby(["session_key", "driver_number"], as_index=False)
        .last()[["session_key", "driver_number", "price"]]
    )
    # group=("session_key",) collapses fills from scattered minutes into one
    # book per race. devig's own gates decide whether that book is complete
    # enough to normalise against.
    anchors = devig(anchors, group=("session_key",))
    anchors = anchors[["session_key", "driver_number", "p_market"]].rename(
        columns={"p_market": "p_market_prerace"}
    )
    return out.merge(anchors, on=["session_key", "driver_number"], how="left")


def add_target(df: pd.DataFrame, inputs: RaceInputs) -> pd.DataFrame:
    """won = 1 for the race winner on every one of that driver's laps.

    The only column allowed to know the outcome, and it is derived from the
    full race on purpose. The replay never calls this.
    """
    out = df.copy()
    position = inputs.position
    if position.empty:
        out["won"] = pd.NA
        return out
    final = position.sort_values("date").groupby("driver_number", as_index=False).last()
    winners = final.loc[final["position"] == 1, "driver_number"]
    out["won"] = out["driver_number"].isin(set(winners)).astype("int64")
    return out


def build_race(
    inputs: RaceInputs,
    prior: pd.Series,
    global_prior: float,
    with_target: bool = True,
) -> pd.DataFrame:
    """Every feature for one race. The replay calls this with truncated inputs."""
    df = lap_frame(inputs)
    df = add_race_state(df, inputs)
    df = add_tyre_state(df, inputs)
    df = add_interruptions(df, inputs)
    df = add_prior(df, inputs, prior, global_prior)
    df = add_market(df, inputs)
    df = add_prerace_market(df, inputs)
    if with_target:
        df = add_target(df, inputs)
    return df.sort_values(["lap_number", "driver_number"], ignore_index=True)


def build(session_keys: list[int] | None = None, seasons=None) -> pd.DataFrame:
    """Full feature table; writes to the `features` table in DuckDB."""
    from fairlap.transform.race_inputs import load

    con = db.connect()
    try:
        sessions = race_sessions(con, seasons)
        if session_keys:
            sessions = sessions[sessions["session_key"].isin(session_keys)]
        prior, global_prior = stop_count_prior(con)
        market = market_fills(con, sessions["session_key"].tolist())

        frames = []
        for _, row in sessions.iterrows():
            inputs = load(int(row["session_key"]), con=con, market=market)
            frames.append(build_race(inputs, prior, global_prior))
        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        _write(con, out, replace_all=not session_keys)
        return out
    finally:
        con.close()


def _write(con, out: pd.DataFrame, replace_all: bool) -> None:
    """Persist the feature table.

    A full build replaces it outright. A build restricted to some races
    replaces only those races: rebuilding one session used to drop the table
    and leave the archive holding that session alone.
    """
    exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'features'"
    ).fetchone()
    if replace_all or not exists:
        con.execute("DROP TABLE IF EXISTS features")
        con.register("_features", out)
        con.execute("CREATE TABLE features AS SELECT * FROM _features")
        con.unregister("_features")
        return
    db.upsert(con, "features", out, key=KEY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the lap-level feature table")
    parser.add_argument("--session-key", type=int, nargs="*", default=None)
    parser.add_argument("--seasons", type=int, nargs="*", default=None)
    args = parser.parse_args()
    out = build(args.session_key, args.seasons)
    print(len(out), "rows written")
