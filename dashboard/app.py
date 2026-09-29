"""Streamlit: pick a race, watch model and market win probabilities diverge.

Reads the DuckDB file read-only. No fitting, no ingestion here -- if the
dashboard needs a number the pipeline does not already store, that number
belongs in a table, not in this file.

Every query opens and closes its own read-only connection. DuckDB allows one
writer or many readers per file, so a connection held open for the life of the
app would make `make replay` and `make eval` fail while the dashboard is up.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Streamlit Community Cloud installs dashboard/requirements.txt and not the
# package itself, so src/ goes on the path by hand. A no-op locally under uv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairlap import db  # noqa: E402
from fairlap.config import DASHBOARD_DB_PATH, DUCKDB_PATH, FEATURED_SESSION_KEY  # noqa: E402
from fairlap.eval.calibration import expected_calibration_error  # noqa: E402

# The full database when it exists, otherwise the committed snapshot from
# `make dashboard-db`. A fresh clone and the hosted app only have the snapshot.
DB_PATH = DUCKDB_PATH if DUCKDB_PATH.exists() else DASHBOARD_DB_PATH

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MODEL_LABELS = {
    "p_market": "market (de-vigged)",
    "p_gbm": "GBM",
    "p_logistic": "logistic",
    "p_position_rate": "baseline A: position rate",
    "p_frozen_prerace": "baseline B: frozen closing line",
}
ZOOM_MAX = 0.15


@st.cache_data(show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    con = db.connect(DB_PATH, read_only=True)
    try:
        return con.execute(sql, list(params)).fetchdf()
    finally:
        con.close()


def has_table(name: str) -> bool:
    tables = query("SELECT table_name FROM information_schema.tables")
    return name in set(tables["table_name"])


def load_races() -> pd.DataFrame:
    """Races that have been replayed, newest first."""
    if not has_table("replay_predictions"):
        return pd.DataFrame()
    return query(
        """
        SELECT r.session_key, r.model, s.year, s.circuit_short_name, s.date_start,
               COUNT(*) AS rows, COUNT(r.p_market) AS priced_rows
        FROM replay_predictions r
        JOIN raw_sessions s USING (session_key)
        GROUP BY ALL
        ORDER BY s.year DESC, s.date_start DESC
        """
    )


def race_frame(session_key: int, model: str) -> pd.DataFrame:
    return query(
        """
        SELECT r.lap_number, r.driver_number, r.p, r.p_market, r.position,
               COALESCE(d.name_acronym, CAST(r.driver_number AS VARCHAR)) AS driver
        FROM replay_predictions r
        LEFT JOIN raw_drivers d USING (session_key, driver_number)
        WHERE r.session_key = ? AND r.model = ?
        ORDER BY r.lap_number, r.driver_number
        """,
        (session_key, model),
    )


def interrupted_laps(session_key: int) -> pd.DataFrame:
    """Laps under SC, VSC or red flag. Each is knowable at the lap it is drawn on."""
    return query(
        """
        SELECT lap_number,
               CASE WHEN BOOL_OR(red_flag_active) THEN 'red flag'
                    WHEN BOOL_OR(sc_active) THEN 'safety car'
                    ELSE 'VSC' END AS condition
        FROM features
        WHERE session_key = ?
        GROUP BY lap_number
        HAVING BOOL_OR(sc_active) OR BOOL_OR(vsc_active) OR BOOL_OR(red_flag_active)
        """,
        (session_key,),
    )


def win_prob_chart(
    df: pd.DataFrame, drivers: list[str], upto: int, total_laps: int, sc: pd.DataFrame
) -> alt.Chart:
    """Model vs market lines over laps, with race-control events marked."""
    part = df[df["driver"].isin(drivers) & (df["lap_number"] <= upto)]
    long = part.melt(
        id_vars=["lap_number", "driver", "position"],
        value_vars=["p", "p_market"],
        var_name="source",
        value_name="prob",
    )
    long["source"] = long["source"].map({"p": "model", "p_market": "market"})

    x = alt.X("lap_number:Q", title="lap", scale=alt.Scale(domain=[1, total_laps]))
    color = alt.Color(
        "driver:N",
        scale=alt.Scale(domain=drivers, range=SERIES[: len(drivers)]),
        title="driver",
        legend=alt.Legend(orient="bottom"),
    )
    dash = alt.StrokeDash(
        "source:N",
        scale=alt.Scale(domain=["model", "market"], range=[[1, 0], [5, 3]]),
        title="source",
        legend=alt.Legend(orient="bottom"),
    )
    # A gap in a market line is a lap with no fresh fill. It is drawn as a gap
    # on purpose: joining across it would draw a price nobody traded.
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("prob:Q", title="win probability", scale=alt.Scale(domain=[0, 1])),
            color=color,
            strokeDash=dash,
            detail="source:N",
        )
    )
    points = (
        alt.Chart(long.dropna(subset=["prob"]))
        .mark_circle(size=40, opacity=0)
        .encode(
            x=x,
            y="prob:Q",
            color=color,
            tooltip=[
                alt.Tooltip("driver:N"),
                alt.Tooltip("lap_number:Q", title="lap"),
                alt.Tooltip("source:N"),
                alt.Tooltip("prob:Q", title="probability", format=".3f"),
                alt.Tooltip("position:Q", format=".0f"),
            ],
        )
    )
    layers = [lines, points]
    sc = sc[sc["lap_number"] <= upto]
    if not sc.empty:
        bands = (
            alt.Chart(sc.assign(lap_end=sc["lap_number"] + 1))
            .mark_rect(opacity=0.12, color="#888888")
            .encode(
                x="lap_number:Q",
                x2="lap_end:Q",
                tooltip=[alt.Tooltip("lap_number:Q", title="lap"), "condition:N"],
            )
        )
        layers.insert(0, bands)
    return alt.layer(*layers).properties(height=420)


def market_ink() -> str:
    # The market is the benchmark, so it is drawn in the page's text colour
    # rather than a series hue -- which means it has to follow the theme.
    theme = getattr(getattr(st, "context", None), "theme", None)
    return "#ffffff" if getattr(theme, "type", None) == "dark" else "#0b0b0b"


def calibration_chart(table: pd.DataFrame, hi: float) -> alt.Chart:
    t = table.assign(label=table["pred"].map(MODEL_LABELS).fillna(table["pred"]))
    order = [MODEL_LABELS[c] for c in MODEL_LABELS if c in set(table["pred"])]
    ink = market_ink()
    colors = [ink if o == MODEL_LABELS["p_market"] else None for o in order]
    it = iter(SERIES)
    colors = [c or next(it) for c in colors]
    diag = (
        alt.Chart(pd.DataFrame({"v": [0, hi]}))
        .mark_line(color="#999999", strokeDash=[4, 4], strokeWidth=1)
        .encode(x="v:Q", y="v:Q")
    )
    curves = (
        alt.Chart(t)
        .mark_line(point=alt.OverlayMarkDef(size=40, clip=True), strokeWidth=2, clip=True)
        .encode(
            x=alt.X(
                "mean_pred:Q", title="mean predicted probability", scale=alt.Scale(domain=[0, hi])
            ),
            y=alt.Y("observed:Q", title="observed win rate", scale=alt.Scale(domain=[0, hi])),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=order, range=colors),
                title="forecaster",
                # Below the plot: in a narrow column a side legend takes the
                # whole width and the plot area collapses to nothing.
                legend=alt.Legend(orient="bottom", columns=2),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="forecaster"),
                alt.Tooltip("bin:Q"),
                alt.Tooltip("n:Q", title="rows"),
                alt.Tooltip("mean_pred:Q", format=".3f"),
                alt.Tooltip("observed:Q", format=".3f"),
            ],
        )
    )
    return alt.layer(diag, curves).properties(height=380)


def race_tab() -> None:
    races = load_races()
    if races.empty:
        st.info("No race has been replayed yet. Run `make replay SESSION_KEY=9693` first.")
        return

    races["label"] = (
        races["year"].astype(str) + " " + races["circuit_short_name"] + " (" + races["model"] + ")"
    )
    featured = races.index[races["session_key"] == FEATURED_SESSION_KEY]
    start = races.index.get_loc(featured[0]) if len(featured) else 0
    pick = st.selectbox("race", races["label"].tolist(), index=start)
    row = races[races["label"] == pick].iloc[0]
    session_key, model = int(row["session_key"]), str(row["model"])

    df = race_frame(session_key, model)
    total_laps = int(df["lap_number"].max())
    # The drivers worth looking at are the ones either side rated at some
    # point, not the final order -- picking by the result would be hindsight.
    peak = df.groupby("driver")[["p", "p_market"]].max().max(axis=1)
    default = peak.sort_values(ascending=False).head(3).index.tolist()
    drivers = st.multiselect(
        "drivers (up to 4)", sorted(df["driver"].unique()), default=default, max_selections=4
    )
    if not drivers:
        return

    c1, c2 = st.columns([4, 1])
    upto = c1.slider("replay up to lap", 1, total_laps, total_laps)
    speed = c2.number_input("seconds per lap", 0.05, 2.0, 0.15, step=0.05)
    play = c2.button("play from lap 1")

    sc = interrupted_laps(session_key)
    slot = st.empty()
    if play:
        for lap in range(1, total_laps + 1):
            slot.altair_chart(win_prob_chart(df, drivers, lap, total_laps, sc), width="stretch")
            time.sleep(speed)
    else:
        slot.altair_chart(win_prob_chart(df, drivers, upto, total_laps, sc), width="stretch")
    st.caption(
        "Solid: model. Dashed: de-vigged market, from a strictly backward as-of join; a gap "
        "means no fill within the staleness tolerance. Grey bands: SC / VSC / red flag. "
        "Market values here are de-vigged across the drivers priced that minute and are not "
        "renormalised to the scored subset, so they differ slightly from the eval inputs."
    )

    lap_now = df[df["lap_number"] == upto].sort_values("p", ascending=False)
    st.dataframe(
        lap_now[["driver", "position", "p", "p_market"]].rename(
            columns={"p": f"{model} p", "p_market": "market p"}
        ),
        hide_index=True,
        width="stretch",
    )


def calibration_tab() -> None:
    if not has_table("eval_calibration"):
        st.info("No eval tables in DuckDB yet. Run `make eval`.")
        return
    table = query("SELECT * FROM eval_calibration")
    left, right = st.columns(2)
    left.markdown("**Full range**")
    left.altair_chart(calibration_chart(table, 1.0), width="stretch")
    right.markdown(f"**Low-probability region (below {ZOOM_MAX})**")
    right.altair_chart(calibration_chart(table, ZOOM_MAX), width="stretch")
    st.caption(
        "Ten equal-count bins per forecaster, 2025-26 held-out rows only. "
        "Above the diagonal means drivers win more often than predicted."
    )
    ece = expected_calibration_error(table).rename("expected calibration error").reset_index()
    ece["pred"] = ece["pred"].map(MODEL_LABELS).fillna(ece["pred"])
    st.dataframe(ece, hide_index=True)


def breakdown_tab() -> None:
    if not has_table("eval_overall"):
        st.info("No eval tables in DuckDB yet. Run `make eval`.")
        return
    model = st.selectbox("model", ["gbm", "logistic", "position_rate", "frozen_prerace"])
    parts = [query("SELECT 'overall' AS breakdown, 'all' AS grp, * FROM eval_overall")]
    for name, col in (("phase", "phase"), ("condition", "condition"), ("season", "season")):
        parts.append(
            query(
                f"SELECT '{name}' AS breakdown, CAST({col} AS VARCHAR) AS grp, "
                f"* EXCLUDE ({col}) FROM eval_by_{name}"
            )
        )
    table = pd.concat(parts, ignore_index=True)
    table = table[table["model"] == model]
    st.dataframe(
        table[
            [
                "breakdown",
                "grp",
                "n",
                "races",
                "brier_model",
                "brier_market",
                "brier_diff",
                "ci_lo",
                "ci_hi",
                "verdict",
            ]
        ].rename(columns={"grp": "group", "n": "rows"}),
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.4f")
            for c in ("brier_model", "brier_market", "brier_diff", "ci_lo", "ci_hi")
        },
    )
    st.caption(
        "Brier difference is model minus market, so positive means the market wins. The 95% "
        "CI resamples whole races. The verdict reads the CI, not the point estimate."
    )


def coverage_tab() -> None:
    if has_table("eval_retention"):
        st.markdown("**Retention per season**: what fraction of model rows the market could score")
        st.dataframe(query("SELECT * FROM eval_retention"), hide_index=True)
    per_race = query(
        """
        SELECT s.year, s.circuit_short_name AS circuit, f.session_key,
               COUNT(*) AS model_rows,
               COUNT(f.p_market) AS priced_rows,
               COUNT(f.p_market) / COUNT(*) AS priced_share,
               COUNT(DISTINCT f.driver_number) FILTER (WHERE f.p_market IS NOT NULL)
                   AS priced_drivers,
               BOOL_OR(f.p_market_prerace IS NOT NULL) AS has_prerace_anchor
        FROM features f
        JOIN raw_sessions s USING (session_key)
        GROUP BY ALL
        HAVING COUNT(f.p_market) > 0
        ORDER BY priced_share
        """
    )
    st.markdown(
        f"**Per race**: {len(per_race)} races have at least one priced row. Sorted thinnest first."
    )
    st.dataframe(
        per_race,
        hide_index=True,
        width="stretch",
        column_config={"priced_share": st.column_config.ProgressColumn(min_value=0, max_value=1)},
    )
    st.caption(
        "A row is priced only if that driver's market had a fill within the staleness "
        "tolerance before the lap ended. Priced rows skew toward eventful laps."
    )


def main() -> None:
    st.set_page_config(page_title="Fair Lap", layout="wide")
    st.title("Fair Lap")
    st.caption("Lap-by-lap F1 win probability vs. Polymarket in-race odds")
    if DB_PATH == DASHBOARD_DB_PATH:
        st.sidebar.caption(f"Reading the committed snapshot, {DB_PATH.name}.")
    if st.sidebar.button("reload from DuckDB"):
        query.clear()
    tabs = st.tabs(["race replay", "calibration", "where each side wins", "coverage"])
    with tabs[0]:
        race_tab()
    with tabs[1]:
        calibration_tab()
    with tabs[2]:
        breakdown_tab()
    with tabs[3]:
        coverage_tab()


if __name__ == "__main__":
    main()
