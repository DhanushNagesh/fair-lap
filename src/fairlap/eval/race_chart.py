"""One race, model vs de-vigged market over laps. The README's first chart.

Reads the held-out GBM predictions (fold -2), not `replay_predictions`, so any
2025-26 race can be drawn without replaying it first. The two are the same
numbers: tests/test_replay.py checks a replay reproduces the holdout rows
exactly.

    uv run python -m fairlap.eval.race_chart --session-key 9947 --out data/race_9947.png
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from fairlap import db

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def load_race(session_key: int, model: str = "gbm") -> pd.DataFrame:
    con = db.connect(read_only=True)
    try:
        return con.execute(
            """
            SELECT f.lap_number, f.driver_number, f.position, f.stops_made,
                   f.sc_active, f.vsc_active, f.red_flag_active, f.p_market, p.p,
                   COALESCE(d.name_acronym, CAST(f.driver_number AS VARCHAR)) AS driver,
                   s.year, s.circuit_short_name
            FROM predictions p
            JOIN features f USING (session_key, lap_number, driver_number)
            JOIN raw_sessions s USING (session_key)
            LEFT JOIN raw_drivers d USING (session_key, driver_number)
            WHERE p.session_key = ? AND p.model = ? AND p.fold = -2
            ORDER BY f.lap_number, f.driver_number
            """,
            [session_key, model],
        ).fetchdf()
    finally:
        con.close()


def load_penalties(session_key: int) -> pd.DataFrame:
    con = db.connect(read_only=True)
    try:
        return con.execute(
            """
            SELECT lap_number, message FROM raw_race_control
            WHERE session_key = ? AND message LIKE 'FIA STEWARDS:%PENALTY FOR CAR%'
              AND message NOT LIKE '%PENALTY SERVED%'
            ORDER BY date
            """,
            [session_key],
        ).fetchdf()
    finally:
        con.close()


def plot_race(df: pd.DataFrame, penalties: pd.DataFrame, n_drivers: int = 3, out_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    laps = np.arange(1, int(df["lap_number"].max()) + 1)
    # Drivers either side rated highly at some point, same rule as the
    # dashboard. Picking by the final result would be hindsight.
    peak = df.groupby("driver")[["p", "p_market"]].max().max(axis=1)
    drivers = peak.sort_values(ascending=False).head(n_drivers).index.tolist()

    fig, ax = plt.subplots(figsize=(11, 5.2))
    flags = df.groupby("lap_number")[["sc_active", "vsc_active", "red_flag_active"]].any()
    for lap, row in flags.iterrows():
        if row.any():
            shade = "0.72" if row["sc_active"] or row["red_flag_active"] else "0.86"
            ax.axvspan(lap - 0.5, lap + 0.5, color=shade, alpha=0.5, linewidth=0, zorder=0)

    for driver, color in zip(drivers, SERIES, strict=False):
        d = df[df["driver"] == driver].set_index("lap_number").reindex(laps)
        ax.plot(laps, d["p"], color=color, linewidth=2.2, label=f"{driver} model")
        # Reindexing leaves NaN where the market had no fresh fill, and
        # matplotlib breaks the line there instead of drawing a price nobody traded.
        ax.plot(
            laps,
            d["p_market"],
            color=color,
            linewidth=1.8,
            linestyle=(0, (5, 3)),
            label=f"{driver} market",
        )
        stops = d.index[d["stops_made"].diff() > 0]
        ax.scatter(
            stops,
            d.loc[stops, "p"],
            marker="v",
            s=55,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

    for _, pen in penalties.iterrows():
        m = re.search(r"CAR \d+ \((\w+)\)", pen["message"])
        if m is None or m.group(1) not in drivers:
            continue
        lap = int(pen["lap_number"])
        ax.axvline(lap, color="0.2", linewidth=1, linestyle=":", zorder=1)
        ax.annotate(
            f"{m.group(1)} gets a 10s penalty",
            xy=(lap, 0.5),
            xytext=(lap + 1.5, 0.5),
            fontsize=9,
            color="0.2",
            va="center",
        )

    first = df.iloc[0]
    ax.set_title(
        f"{first['year']} {first['circuit_short_name']}: GBM vs de-vigged Polymarket, lap by lap",
        loc="left",
    )
    ax.set_xlim(0.5, laps[-1] + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("lap")
    ax.set_ylabel("win probability")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([], [], color=c, linewidth=2.2, label=d)
        for d, c in zip(drivers, SERIES, strict=False)
    ]
    handles += [
        Line2D([], [], color="0.3", linewidth=2.2, label="model"),
        Line2D([], [], color="0.3", linewidth=1.8, linestyle=(0, (5, 3)), label="market"),
        Line2D([], [], color="0.3", marker="v", linestyle="", label="pit stop"),
        Patch(color="0.72", alpha=0.5, label="safety car"),
        Patch(color="0.86", alpha=0.5, label="VSC"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session-key", type=int, required=True)
    parser.add_argument("--model", default="gbm")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    df = load_race(args.session_key, args.model)
    if df.empty:
        raise SystemExit(f"no held-out predictions for session {args.session_key}")
    plot_race(df, load_penalties(args.session_key), out_path=args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
